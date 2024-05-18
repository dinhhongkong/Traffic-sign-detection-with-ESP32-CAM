#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <WiFi.h>
#include <WebSocketsServer.h>

#include "esp_camera.h"
#include "ImageProcessing.h"
#include "ConfigCamera.h"

#include "bbgt_model.h"

const float detectionThreshold = 0.7;

// Define two tasks for Blink & AnalogRead.
void TaskBlink( void *pvParameters );
void TaskAnalogRead( void *pvParameters );
void TaskRunServerSocket(void *pvParameters);


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


const char* classNames[] = {
  "Gioi han toc do (30km/h)",
  "Giao nhau voi duong uu tien",
  "Cong trinh dang thi cong"
};

const String indexAns[] = {
  "0","1","2"
};

// Function to print prediction result
void printResult() {

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

    // send Multicast
    for (uint8_t i = 0; i < webSocket.connectedClients(); ++i) {
      webSocket.sendTXT(i, indexAns[predictedClass] + " Probability: " + String(maxProbability * 100, 2) );
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

void initTensorFlowLite() {
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
  

}

void initWifiServerSocket() {
  WiFi.softAP(ssid, password); // Bắt đầu phát WiFi với tên mạng và mật khẩu được chỉ định
  IPAddress myIP = WiFi.softAPIP(); // Lấy địa chỉ IP của ESP32-CAM trong mạng WiFi mà nó phát
  Serial.print("AP IP address: ");
  Serial.println(myIP);
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
}


void setup() {
  Serial.begin(115200);
  
  camera_config_t config;
  configCamera(config);

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // Initialize TensorFlow Lite
  initTensorFlowLite();

  //Initialize wifi server socket
  initWifiServerSocket();



  // Tasks in RTOS
  uint32_t blink_delay = 1000; // Delay between changing state on LED pin
  
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
    ,  2  // Priority
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




