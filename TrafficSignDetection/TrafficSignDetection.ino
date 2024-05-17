#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_camera.h"


#include <WiFi.h>
#include <WebSocketsServer.h>

#define ASCII_CHARS " .:-=+*#%@"

#include "bbgt_model.h"

const float detectionThreshold = 0.4;

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#if CONFIG_FREERTOS_UNICORE
#define ARDUINO_RUNNING_CORE 0
#else
#define ARDUINO_RUNNING_CORE 1
#endif

#define ANALOG_INPUT_PIN A0

#ifndef LED_BUILTIN
  #define LED_BUILTIN 13 // Specify the on which is your LED
#endif

// Define two tasks for Blink & AnalogRead.
void TaskBlink( void *pvParameters );
void TaskAnalogRead( void *pvParameters );
void TaskRunServerSocket(void *pvParameters);
TaskHandle_t analog_read_task_handle; // You can (don't have to) use this to be able to manipulate a task from somewhere else.

// 4 for flash led or 33 for normal led
#define LED_GPIO_NUM       4

// Constants for image processing
constexpr int kImageWidth = 32;
constexpr int kImageHeight = 32;
constexpr int kImageChannels = 1; // Grayscale
uint8_t image_data[kImageWidth * kImageHeight * kImageChannels];


// TensorFlow Lite variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
constexpr int kTensorArenaSize = 32*1024;
uint8_t tensor_arena[kTensorArenaSize];


// Constants for Wifi Server
const char* ssid = "Traffic sign detection";
const char* password = "1234567890";
WebSocketsServer webSocket = WebSocketsServer(80);



// Function to resize image to 32x32
void resize_image_to_32x32(uint8_t* input, uint8_t* output, int inputWidth, int inputHeight) {
  float scaleWidth = inputWidth / (float)kImageWidth;
  float scaleHeight = inputHeight / (float)kImageHeight;

  for (int y = 0; y < kImageHeight; y++) {
    for (int x = 0; x < kImageWidth; x++) {
      int srcX = (int)(x * scaleWidth);
      int srcY = (int)(y * scaleHeight);
      srcX = min(srcX, inputWidth - 1);
      srcY = min(srcY, inputHeight - 1);
      int inputIndex = (srcY * inputWidth) + srcX;
      int outputIndex = (y * kImageWidth) + x;
      output[outputIndex] = input[inputIndex];
    }
  }
}


// Function to preprocess image
void preprocess_image(uint8_t* input, float* output, int width, int height) {
    uint8_t grayscale_img[width * height];
    uint8_t equalized_img[width * height];

    // Convert image to grayscale
    for (int i = 0; i < width * height; i++) {
        grayscale_img[i] = input[i];
    }

    // Equalize histogram
    int hist[256] = {0};
    for (int i = 0; i < width * height; i++) {
        hist[grayscale_img[i]]++;
    }
    int cumulative = 0;
    int minCumulative = width * height;
    for (int i = 0; i < 256; i++) {
        cumulative += hist[i];
        if (cumulative > 0 && cumulative < minCumulative) {
            minCumulative = cumulative;
        }
    }
    float scale = 255.0 / (width * height - minCumulative);
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += hist[i];
        hist[i] = round((sum - minCumulative) * scale);
    }
    for (int i = 0; i < width * height; i++) {
        equalized_img[i] = hist[grayscale_img[i]];
    }

    // Normalize the image and copy to output
    for (int i = 0; i < width * height; i++) {
        output[i] = equalized_img[i] / 255.0f;
    }
}

// Function to convert pixel value to ASCII character
char pixelToAscii(uint8_t pixelValue) {
  // ASCII characters arranged from darkest to lightest
  static const char asciiChars[] = " .:-=+*#";
  // Determine ASCII index based on pixel value
  int asciiIndex = map(pixelValue, 0, 255, 0, strlen(asciiChars) - 1);
  // Return corresponding ASCII character
  return asciiChars[asciiIndex];
}


// Function to display image as ASCII art with smaller size
void displayAsciiArt(camera_fb_t *fb) {
  // Display image as ASCII art in Serial Monitor
  for (int y = 0; y < fb->height; y += 8) {
    for (int x = 0; x < fb->width; x += 4) {
      // Calculate average pixel value in 8x8 block
      int sum = 0;
      for (int dy = 0; dy < 8; dy++) {
        for (int dx = 0; dx < 4; dx++) {
          int index = (y + dy) * fb->width + (x + dx);
          sum += fb->buf[index];
        }
      }
      int avg = sum / 32;

      // Convert average pixel value to ASCII character
      char asciiChar = pixelToAscii(avg);
      
      // Print ASCII character
      Serial.write(asciiChar);
    }
    Serial.println(); // Newline after each row
  }
}

const char* classNames[] = {
  "Gioi han toc do (30km/h)",
  "Giao nhau voi duong uu tien",
  "Cong trinh dang thi cong"
};

const char* indexAns[] = {
  "0","1","2"
};

// Function to print prediction result
void printResult() {
  Serial.println("Task printResult: ");
  // Get output tensor
  TfLiteTensor* outputTensor = interpreter->output(0);
  
  // Find the predicted class with the highest probability
  int predictedClass = -1;
  float maxProbability = 0.0;
  for (int i = 0; i < outputTensor->dims->data[1]; ++i) {
    float probability = outputTensor->data.f[i];
    if (probability > maxProbability) {
      maxProbability = probability;
      predictedClass = i;
    }
  }


    // Print the result if the probability is above the threshold
  if (maxProbability > detectionThreshold) {
    Serial.print("Predicted Class: ");
    Serial.println(predictedClass);
    Serial.print("Class Name: ");
    Serial.println(classNames[predictedClass]);
    Serial.print("Probability: ");
    Serial.println(maxProbability);
    for (uint8_t i = 0; i < webSocket.connectedClients(); ++i) {
      webSocket.sendTXT(i, indexAns[predictedClass]);
    }
  }
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", num);
      break;
    case WStype_CONNECTED:
      {
        IPAddress ip = webSocket.remoteIP(num);
        Serial.printf("[%u] Connected from %d.%d.%d.%d url: %s\n", num, ip[0], ip[1], ip[2], ip[3], payload);
        
        // Gửi thông điệp chào mừng đến thiết bị kết nối
        webSocket.sendTXT(num, "Chào mừng đến với ESP32-CAM WebSocket Server!");
      }
      break;
  
  }
}


void setup() {
  Serial.begin(115200);
  
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // Initialize TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(bbgt_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  uint32_t blink_delay = 1000; // Delay between changing state on LED pin

  //Init wifi server
  WiFi.softAP(ssid, password); // Bắt đầu phát WiFi với tên mạng và mật khẩu được chỉ định

  IPAddress myIP = WiFi.softAPIP(); // Lấy địa chỉ IP của ESP32-CAM trong mạng WiFi mà nó phát
  Serial.print("AP IP address: ");
  Serial.println(myIP);
  
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);


  xTaskCreatePinnedToCore(
    TaskBlink
    ,  "Task Blink" // A name just for humans
    ,  4096        // The stack size can be checked by calling `uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);`
    , (void*) &blink_delay // Task parameter which can modify the task behavior. This must be passed as pointer to void.
    ,  3  // Priority
    ,  NULL // Task handle is not used here - simply pass NULL
    , 0
    );  

  // This variant of task creation can also specify on which core it will be run (only relevant for multi-core ESPs)
  xTaskCreatePinnedToCore(
    TaskAnalogRead
    ,  "Analog Read"
    ,  2048  // Stack size
    ,  NULL  // When no parameter is used, simply pass NULL
    ,  1  // Priority
    ,  NULL  // With task handle we will be able to manipulate with this task.
    ,  1 // Core on which the task will run
    );

  xTaskCreatePinnedToCore(
    TaskRunServerSocket
    ,  "Run ServerSocket" // A name just for humans
    ,  2048        // The stack size can be checked by calling `uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);`
    ,  NULL // Task parameter which can modify the task behavior. This must be passed as pointer to void.
    ,  2  // Priority
    ,  NULL // Task handle is not used here - simply pass NULL
    ,  1
    );  


  Serial.printf("chay xong setup()");
}

void loop() {


}



/*--------------------------------------------------*/
/*---------------------- Tasks ---------------------*/
/*--------------------------------------------------*/
// 
// Define task 1
void TaskBlink(void *pvParameters){  // This is a task.
  uint32_t blink_delay = *((uint32_t*)pvParameters);


  for(;;)
  {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      return;
    }

    // Resize image to 32x32
    resize_image_to_32x32(fb->buf, image_data, fb->width, fb->height);


    // Preprocess image data
    preprocess_image(image_data, model_input->data.f, kImageWidth, kImageHeight);


    // Invoke interpreter
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on x_val");
      return;
    }

    // Display image as ASCII art
    displayAsciiArt(fb);

    // Print the results directly
    Serial.print("class 1(ghtd 30k/h): ");
    Serial.println(model_output->data.f[0]);

    Serial.print("class 2(giao duong uu tien): ");
    Serial.println(model_output->data.f[1]);

    Serial.print("class 3(ct dang thi cong): ");
    Serial.println(model_output->data.f[2]);


  // Print prediction result
    // printResult();
  
    Serial.println(". ");
    Serial.println(".. ");
    Serial.println();
    Serial.println();

  // Cleanup
    esp_camera_fb_return(fb);
//  Serial.print("Free heap: ");
//  Serial.println(ESP.getFreeHeap());
//  UBaseType_t stackRemaining = uxTaskGetStackHighWaterMark(NULL);
//Serial.print("Remaining stack space: ");
//Serial.println(stackRemaining);

    // delay(3000);
    vTaskDelay(3000);

  }

}
// define task2
void TaskAnalogRead(void *pvParameters){  // This is a task.
  (void) pvParameters;
  for (;;){
      // Print prediction result
    printResult();

  }
}

void TaskRunServerSocket(void *pvParameters){ 
  (void) pvParameters;
  while(true) {
    webSocket.loop(); 
  }
  
}




