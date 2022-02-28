/*デイジーチェン接続 
Arduino11番ピン（MOSI）- 6番ピン(SDI) L6470(一台目) 3番ピン(SDO) - (SDI)L6470(二台目)(SDO) - 12番ピン（MISO) Arduino
SCK 共通
SS - CS共通
*/
#include <SPI.h>
#include <stdio.h>

// ピン定義。
#define PIN_SPI_MOSI 11
#define PIN_SPI_MISO 12
#define PIN_SPI_SCK 13
#define PIN_SPI_SS 10
#define PIN_BUSY 9
#define PIN_BUSY2 8

void setup()
{
  pinMode(PIN_SPI_MOSI, OUTPUT);
  pinMode(PIN_SPI_MISO, INPUT);
  pinMode(PIN_SPI_SCK, OUTPUT);
  pinMode(PIN_SPI_SS, OUTPUT);
  pinMode(PIN_BUSY, INPUT_PULLUP);
  pinMode(PIN_BUSY2, INPUT_PULLUP);
  SPI.begin();
  SPI.setDataMode(SPI_MODE3);
  SPI.setBitOrder(MSBFIRST);
  Serial.begin(9600);
  digitalWrite(PIN_SPI_SS, HIGH);

  L6470_resetdevice(); //1台目のL6470リセット
  L6470_resetdevice2(); //2台目のL6470リセット
  L6470_setup();  //1台目のL6470を設定 
  L6470_setup2();  //2台目のL6470を設定 
  L6470_getstatus(); //1台目のフラグ解放
  L6470_getstatus2();//2台目のフラグ解放
}

void loop(){
  while (1) {
    String line;   // 受信文字列  
    String data_[4] = {"\0"};

    signed char c;
    //データが存在するときだけ受信処理以降を行う(頭の0が使われる)
    if((c = Serial.read()) != -1){
      line = Serial.readStringUntil('\n');
      char ind = 0;
      char len_ = line.length();
        
      for (char i = 0; i < len_; i++) {
        char tmp = line.charAt(i);
        if ( tmp == ',' ) {
          ind++;
        }
        else data_[ind] += tmp;
      }
  
      if (data_[0] == "remark") {
        L6470_setparam_mark(data_[1].toInt());
        L6470_setparam_mark2(data_[2].toInt());
      }
      else if (data_[0] == "reset"){
        L6470_gomark();
        L6470_gomark2();
      }
      else if (data_[0] == "home"){
        L6470_gohome();
        L6470_gohome2();
      }
      
      else {
        //回転方向,step数
        L6470_move(data_[0].toInt(),data_[1].toInt());//指定方向に連続回転
        L6470_move2(data_[2].toInt(),data_[3].toInt());//指定方向に連続回転
        L6470_busydelay(0); //命令を受け付けるまで待機
        L6470_busydelay2(0); //命令を受け付けるまで待機
        Serial.println("d");        
      }
    }
  }
}

void L6470_setup(){
L6470_setparam_acc(3000); //[R, WS] 加速度default 0x08A (12bit) (14.55*val+14.55[step/s^2])
L6470_setparam_dec(3000); //[R, WS] 減速度default 0x08A (12bit) (14.55*val+14.55[step/s^2])
L6470_setparam_maxspeed(90); //[R, WR]最大速度default 0x041 (10bit) (15.25*val+15.25[step/s])
L6470_setparam_minspeed(0x1200); //[R, WS]最小速度default 0x000 (1+12bit) (0.238*val[step/s])
L6470_setparam_fsspd(0x027); //[R, WR]μステップからフルステップへの切替点速度default 0x027 (10bit) (15.25*val+7.63[step/s])
L6470_setparam_kvalhold(0x28); //[R, WR]停止時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)
L6470_setparam_kvalrun(0x28); //[R, WR]定速回転時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)
L6470_setparam_kvalacc(0x28); //[R, WR]加速時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)
L6470_setparam_kvaldec(0x28); //[R, WR]減速時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)

L6470_setparam_stepmood(6); //ステップモードdefault 0x07 (1+3+1+3bit)
}

void L6470_setup2(){
L6470_setparam_acc2(3000); //[R, WS] 加速度default 0x08A (12bit) (14.55*val+14.55[step/s^2])
L6470_setparam_dec2(3000); //[R, WS] 減速度default 0x08A (12bit) (14.55*val+14.55[step/s^2])
L6470_setparam_maxspeed2(90); //[R, WR]最大速度default 0x041 (10bit) (15.25*val+15.25[step/s])
L6470_setparam_minspeed2(0x1200); //[R, WS]最小速度default 0x000 (1+12bit) (0.238*val[step/s])
L6470_setparam_fsspd2(0x027); //[R, WR]μステップからフルステップへの切替点速度default 0x027 (10bit) (15.25*val+7.63[step/s])
L6470_setparam_kvalhold2(0x28); //[R, WR]停止時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)
L6470_setparam_kvalrun2(0x28); //[R, WR]定速回転時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)
L6470_setparam_kvalacc2(0x28); //[R, WR]加速時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)
L6470_setparam_kvaldec2(0x28); //[R, WR]減速時励磁電圧default 0x29 (8bit) (Vs[V]*val/256)

L6470_setparam_stepmood2(6); //ステップモードdefault 0x07 (1+3+1+3bit)
}


void fulash(){
  long a=L6470_getparam_abspos();
  long b=L6470_getparam_speed();
  long c=L6470_getparam_abspos2();
  long d=L6470_getparam_speed2();
  char str[15];
  snprintf(str,sizeof(str),"1pos=0x%6.6X ",a);
  snprintf(str,sizeof(str),"1spd=0x%5.5X ",b);
  snprintf(str,sizeof(str),"2pos=0x%6.6X ",c);
  snprintf(str,sizeof(str),"2spd=0x%5.5X ",d);
}
