#include <stdio.h>
#include <wiringPi.h>
#include <pcf8591.h>
#include <math.h>
#include <softPwm.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include  <wiringPiI2C.h>

#define		PCF     120

#define uchar unsigned char

#define LedPinRed    24
#define LedPinGreen  1

#define Trig    27
#define Echo    23

#define BuzzerPin    26

#define  RoAPin    22
#define  RoBPin    21
#define  SWPin     7

static volatile int globalCounter = 0 ;

unsigned char flag;
unsigned char Last_RoB_Status;
unsigned char Current_RoB_Status;

int fd1;
int acclX, acclY, acclZ;
int gyroX, gyroY, gyroZ;
double acclX_scaled, acclY_scaled, acclZ_scaled;
double gyroX_scaled, gyroY_scaled, gyroZ_scaled;

int LCDAddr = 0x27;
int BLEN = 1;
int fd;

void ledInit(void)
{
	softPwmCreate(LedPinRed,  0, 100);
	softPwmCreate(LedPinGreen,0, 100);
}

void ledColorSet(uchar r_val, uchar g_val)
{
	softPwmWrite(LedPinRed,   r_val);
	softPwmWrite(LedPinGreen, g_val);
}

void Print(int x)
{
	if (x <200){
			printf("\n***************\n"  );
			printf(  "*Raining *\n"  );
			printf(  "***************\n\n");
       }
	else{
			printf("\n*************\n"  );
			printf(  "*Not Raining!! *\n"  );
			printf(  "*************\n\n");
       }
}

void ultraInit(void)
{
	pinMode(Echo, INPUT);  //回波引脚
	pinMode(Trig, OUTPUT); //触发引脚
}

float disMeasure(void)
{
	struct timeval tv1;
	struct timeval tv2; //存储时间戳信息
	long time1, time2;
    float dis;

	digitalWrite(Trig, LOW);
	delayMicroseconds(2); //触发引脚设置为低电平，并延时2微秒

	digitalWrite(Trig, HIGH); //触发引脚设置为高电平，并延时10微秒
	delayMicroseconds(10);      
	digitalWrite(Trig, LOW);
								
	while(!(digitalRead(Echo) == 1));
	gettimeofday(&tv1, NULL);           //获取当前时间

	while(!(digitalRead(Echo) == 0));
	gettimeofday(&tv2, NULL);           //获取当前时间

	time1 = tv1.tv_sec * 1000000 + tv1.tv_usec;   //微秒级的时间
	time2  = tv2.tv_sec * 1000000 + tv2.tv_usec;

	dis = (float)(time2 - time1) / 1000000 * 34000 / 2;  
    //求出距离

	return dis;
}

void btnISR(void)
{
	globalCounter = 0;
}

void rotaryDeal(void)
{
  Last_RoB_Status = digitalRead(RoBPin);

	while(!digitalRead(RoAPin)){
		Current_RoB_Status = digitalRead(RoBPin);
		flag = 1;
	}

	if(flag == 1){
		flag = 0;
		if((Last_RoB_Status == 0)&&(Current_RoB_Status == 1)){
			globalCounter ++;	
		}
		if((Last_RoB_Status == 1)&&(Current_RoB_Status == 0)){
			globalCounter --;
		}
	}
}

int read_word_2c(int addr)
{
  int val;
  val = wiringPiI2CReadReg8(fd1, addr);
  val = val << 8;
  val += wiringPiI2CReadReg8(fd1, addr+1);
  if (val >= 0x8000)
    val = -(65536 - val);

  return val;
}

void write_word(int data){
	int temp = data;
	if ( BLEN == 1 )
		temp |= 0x08;
	else
		temp &= 0xF7;
	wiringPiI2CWrite(fd, temp);
}

void send_command(int comm){
	int buf;
	// Send bit7-4 firstly
	buf = comm & 0xF0;
	buf |= 0x04;			// RS = 0, RW = 0, EN = 1
	write_word(buf);
	delay(2);
	buf &= 0xFB;			// Make EN = 0
	write_word(buf);

	// Send bit3-0 secondly
	buf = (comm & 0x0F) << 4;
	buf |= 0x04;			// RS = 0, RW = 0, EN = 1
	write_word(buf);
	delay(2);
	buf &= 0xFB;			// Make EN = 0
	write_word(buf);
}

void send_data(int data){
	int buf;
	// Send bit7-4 firstly
	buf = data & 0xF0;
	buf |= 0x05;			// RS = 1, RW = 0, EN = 1
	write_word(buf);
	delay(2);
	buf &= 0xFB;			// Make EN = 0
	write_word(buf);

	// Send bit3-0 secondly
	buf = (data & 0x0F) << 4;
	buf |= 0x05;			// RS = 1, RW = 0, EN = 1
	write_word(buf);
	delay(2);
	buf &= 0xFB;			// Make EN = 0
	write_word(buf);
}

void init(){
	send_command(0x33);	
    // Must initialize to 8-line mode at first
	delay(5);
	send_command(0x32);	// Then initialize to 4-line mode
	delay(5);
	send_command(0x28);	// 2 Lines & 5*7 dots
	delay(5);
	send_command(0x0C);	// Enable display without cursor
	delay(5);
	send_command(0x01);	// Clear Screen
	wiringPiI2CWrite(fd, 0x08);
}

void clear(){
	send_command(0x01);	//clear Screen
}

void write(int x, int y, char data[]){
	int addr, i;
	int tmp;
    // 确保 x、y 在有效范围内
	if (x < 0)  x = 0;
	if (x > 15) x = 15;
	if (y < 0)  y = 0;
	if (y > 1)  y = 1;
    // 计算LCD的地址
	addr = 0x80 + 0x40 * y + x;
    // 发送命令，设置LCD的光标位置
	send_command(addr);
	// 计算数据字符串的长度
	tmp = strlen(data);
    // 遍历数据字符串，逐个发送字符数据
	for (i = 0; i < tmp; i++){
		send_data(data[i]);
	}
}




int main()
{
	int analogVal;
    int i;
    int rainVal;
	int tmp, status;
 	float dis;
	
	if(wiringPiSetup() == -1){
		printf("setup wiringPi failed !");
		return 1;
	}
 
	pcf8591Setup(PCF, 0x48);
    ledInit();
	status = 0;
  	ultraInit();
   	pinMode(BuzzerPin,  OUTPUT);
    digitalWrite(BuzzerPin, HIGH);
    pinMode(SWPin, INPUT);
	pinMode(RoAPin, INPUT);
	pinMode(RoBPin, INPUT);
	pullUpDnControl(SWPin, PUD_UP);

    if(wiringPiISR(SWPin, INT_EDGE_FALLING, &btnISR) < 0){
		fprintf(stderr, "Unable to init ISR\n",strerror(errno));	
		return 1;
	}
	
	int tmpR = 0;
    fd1 = wiringPiI2CSetup (0x68);
    wiringPiI2CWriteReg8 (fd1,0x6B,0x00);//disable sleep mode 
    printf("set 0x6B=%X\n",wiringPiI2CReadReg8 (fd1,0x6B));
    fd = wiringPiI2CSetup(LCDAddr);
	init();
 
  char lightvalue[16]; //存放光敏电阻那边的数字
  char waterValue[16]; //存放雨滴那边的数字
  char chaoshengValue[16]; //存放超声数字
  char licheng[16]; //存放里程
  
  
  	pullUpDnControl(RoAPin, PUD_UP); // 初始时上拉A通道
    pullUpDnControl(RoBPin, PUD_UP); // 初始时上拉B通道
    delay(10);
    wiringPiISR(RoAPin, INT_EDGE_FALLING, &rotaryDeal);
  
  
	while(1) 
	{
		analogVal = analogRead(PCF + 0);
        sprintf(lightvalue,"light:%2d",analogVal);
        write(0, 0, lightvalue);

		delay (100);
   
		ledColorSet(0x00,0xff);   //green
        delay (200);
   
        if(analogVal >=60){
        ledColorSet(0xff,0x00);   //red	
  		delay(500);
     }
   

    rainVal = analogRead(PCF + 1);
    //sprintf(waterValue,"water:%2d",rainVal);
    // write(0, 1, waterValue);
	// delay(200);
    //clear();

		if (rainVal != status)
		{
			Print(rainVal);
			status = rainVal;
		}
   if(rainVal<200){
      write(0, 1, "it's raining!");
      delay(200);
   }
   else{
   write(0, 1, "no rain");
   delay (200);
   }
   clear();

   
    dis = disMeasure();
   sprintf(chaoshengValue,"dis:%2f",dis);
   write(0, 0, chaoshengValue);
   
    if(dis<50){
    digitalWrite(BuzzerPin, HIGH);
    delay(dis*0.5);
    digitalWrite(BuzzerPin, LOW);
    delay(dis*20);
    }
    else{
    digitalWrite(BuzzerPin, HIGH);
    }
    
    //rotaryDeal();
   sprintf(licheng,"licheng:%2d",globalCounter);
   write(0, 1, licheng);
	 delay(200);
   clear();
   
			tmpR = globalCounter;
      delay(200);

   
    gyroX = read_word_2c(0x43);
    gyroY = read_word_2c(0x45);
    gyroZ = read_word_2c(0x47);

    //数据缩放
    gyroX_scaled = gyroX / 131.0;
    gyroY_scaled = gyroY / 131.0;
    gyroZ_scaled = gyroZ / 131.0;
    //检查阈值:40,超过40，表示可能发生了剧烈的运动
    if(abs(gyroX_scaled)>40 || abs(gyroY_scaled)>40 || abs(gyroZ_scaled)>40){
    digitalWrite(BuzzerPin, LOW);
    delay(100);
    }
    else{
    digitalWrite(BuzzerPin, HIGH);
    }
   
	}
	return 0;
}
