#include <Arduino.h>
#include <Wire.h>

void setup() {
  Serial.begin(115200);
  Wire.begin();  // I2C for camera
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);  // Indicate active
  delay(1000);
  digitalWrite(LED_BUILTIN, LOW);
  delay(1000);
}
