#include <SPI.h>
#include <stdio.h>

// ピン定義。
#define arrayLength binary
#define PIN_LED 7

int val = 0;
// フォトトランジスタの電圧を確認するピンはA0
int iPin = A0;
// レーザーの点滅は7番ピンで操作
int oPin = 7;
int i;
int done = 0;
// 点滅の閾値
int thresh = 500;
// 送信信号
String s = "010010000110010101101100011011000110111100100000011101110110111101110010011011000110010000101110";

void setup() {
  Serial.begin(9600);
  pinMode(oPin, OUTPUT);
}

void loop() {
  delay(3000);
  if (done == 0) {
    String get_s = {"\0"};
    Serial.println("start");
    for (i=0;i<s.length();i++){
      String j = String(s[i]);
      // LEDの点滅
      if (j.toInt() == 0) {
        digitalWrite(oPin, LOW);
      } else if (j.toInt() == 1) {
        digitalWrite(oPin, HIGH);
      }
      // フォトトランジスタでのデータ受信
      val = analogRead(iPin);
      Serial.println(val);
      if (val > thresh) {
        get_s += "1";
      } else {
        get_s += "0";
      }
      //delay(10);
      
    }
    Serial.println("end");
    Serial.println(get_s);
    //done = 1;
  }
  digitalWrite(oPin, LOW);
}
