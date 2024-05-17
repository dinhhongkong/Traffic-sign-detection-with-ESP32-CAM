
// Constants for image processing
constexpr int kImageWidth = 32;
constexpr int kImageHeight = 32;
constexpr int kImageChannels = 1; // Grayscale
uint8_t image_data[kImageWidth * kImageHeight * kImageChannels];

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
