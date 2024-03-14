#include <Time.h>
#include <TimeLib.h>
#include <stdlib.h>

#define NSERVOS 2
#define NMOTORS 2
#define DISTART 39
#define DISTOP 53
#define DISTEP 2
#define DOSTART 24
#define DOSTOP 36 // note : 8th digital channel is the busystate now
#define DOSTEP 2
#define NAI 8
#define AISTART 8
#define AISTOP AISTART + NAI - 1
#define MOTORSHIELDPIN 7
#define PULSEPIN 52
#define TRIGGERPIN 39
#define LED 13
#define BUTTON 38
#define DEVICEINFO "Mega_IO"

// Initialize Main Variables 
int begintrigger=0,val=LOW;
int analogData[NAI];
double resettime=pow(2,32);

int pos1 = 0, pos2 = 0, motorid;
int msteps, pin, inout, hilow, mdir;
int timewas=0, state;
int sending =0, busystate =0; 
int i, stepsize = 1, servopos = 0;
int current_angle[2] = {105,80};
char para[9];
char bcheck = 'b';
char mode = 'x';
int AIMAP[16] = {A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15};

// Define various ADC prescaler
const unsigned char PS_16 = (1 << ADPS2);
const unsigned char PS_32 = (1 << ADPS2) | (1 << ADPS0);
const unsigned char PS_64 = (1 << ADPS2) | (1 << ADPS1);
const unsigned char PS_128 = (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);

// FIRST FUNCTION CALLED
void setup() {   

  //setup the pulse generator as output
  pinMode(PULSEPIN, OUTPUT);
  pinMode(TRIGGERPIN, INPUT);

  // Setup digital output
  pinMode(13, OUTPUT);// turn on the emitters
  digitalWrite(13,LOW);
  for (i=DOSTART; i<=DOSTOP; i+=DOSTEP) {
    pinMode(i, OUTPUT); 
    digitalWrite(i, HIGH);
  }

  // set up the ADC
  ADCSRA &= ~PS_128;  // remove bits set by Arduino library
  // you can choose a prescaler from above.
  // PS_16, PS_32, PS_64 or PS_128
  ADCSRA |= PS_32;    // set our own prescaler to 64 

  // Setup digital input
  for (i=DISTART; i<=DISTOP; i+=DISTEP) pinMode(i, INPUT);  

  //digitalWrite (BUTTON, HIGH);  // internal pull-up resistor
  //attachInterrupt (0, pinChange, CHANGE);

  // Start Serial Port Communication at 115200 bps
  Serial.begin(115200);        
}

// MAIN LOOP ===============================================================================
void loop() {

  if(Serial.available()>10) { 
    commandread();
    // Check which command has arrived    
    if (mode=='a')  { sending = 1;       timewas = micros(); }
    if (mode=='h')  { sending = 0;       mode = 'x';  }
    if (mode=='d')  { pinset();          mode = 'x';  }
//    if (mode=='s')  { servorotate();     mode = 'x'; }
//    if (mode=='m')  { motormove();       mode = 'x'; } 
    if (mode=='p')  { pulsegenerator();  mode = 'x'; }
    if (mode=='t')  { triggersend();     mode = 'x'; }
    if (mode=='i')  { Serial.write(DEVICEINFO); Serial.write('\n'); mode = 'x';}
    busystate = 0;
  }

  if (sending==1) datasend();
}

// Send data out via the serial connection
int datasend() { 
  int cPin, timediff, timenow, digitalreadval=0;
  // Read all Analog Sensors
  for (cPin=0; cPin<NAI; cPin++) {
    analogData[cPin] = analogRead(AIMAP[AISTART + cPin]);
    analogData[cPin] = map(analogData[cPin], 0, 1023, 0, 255); 
  }
  // Read all Digital Sensors
  for (cPin=DISTART; cPin<=DISTOP; cPin+=DISTEP) {
    if (digitalRead(cPin)==HIGH) 
      bitSet(digitalreadval,(cPin-DISTART)/DISTEP);
  }
  // Report whether the Arduino is busy with motor tasks
  if (busystate==HIGH) bitSet(digitalreadval,7);
  timenow=micros();
  timediff=timenow-timewas;
  if (timediff<0)  timediff = (timenow + resettime - timewas);
  timewas=timenow;
  //timediff=timediff/10;
  for (cPin=0; cPin<NAI; cPin++) Serial.write(analogData[cPin]);
  //Serial.write(analogData8,NAI);
  Serial.write(digitalreadval);
  //for (ii=0; ii<1; ii++) Serial.print('a');
  //Serial.write(busystate);
  Serial.write(timediff/1000);
  Serial.write((timediff/10)%100);
  Serial.write(254);
  Serial.write(255);
  return 1;
}

// Read the command from the serial line
void commandread(){
  int ii;
  bcheck= (char) Serial.read();
  if (bcheck=='b') {
    mode = (char) Serial.read();
    for (ii=0; ii<9; ii++) para[ii] = (char) Serial.read();
    //Serial.println(mode);
  }
  //Serial.println(Serial.available());
}

// Ste a given input/output pin   
void pinset() {
  int PinNum = (para[0]-'0')*10 + (para[1]-'0')*1;
  if (para[2]=='1')  digitalWrite(PinNum, HIGH);
  if (para[2]=='0')  digitalWrite(PinNum, LOW);      
}

void pulsegenerator()  {
  int period;
  unsigned long modTime;
  unsigned long tstart;
  int pulsedur = 200; 
  int rundirectly = 0;
  period = (para[0]-'0')*100000+(para[1]-'0')*10000+(para[2]-'0')*1000+(para[3]-'0')*100+(para[4]-'0')*10+(para[5]-'0')*1;
  rundirectly = (int) (para[6]-'0')*1;
  val = HIGH; 
    
  if (rundirectly==2) {
   digitalWrite(LED,HIGH);
   val = digitalRead(TRIGGERPIN);
   while (val==LOW) {
     if (sending) {datasend();}
     val = digitalRead(TRIGGERPIN); 
   }
  }
  
  tstart = micros();
  while (val==HIGH){
    modTime = (micros()-tstart)%period;
    // WAIT UNTIL PULSE SHOULD START 
    while (modTime > pulsedur)
      modTime = (micros()-tstart)%period;
    digitalWrite(PULSEPIN, HIGH);
    // WAIT UNTIL PULSE SHOULD CEASE
    while (modTime <= pulsedur) 
      modTime = (micros() - tstart)%period;
    digitalWrite(PULSEPIN, LOW);
    // CHECK WHETHER TO STOP
    val = digitalRead(TRIGGERPIN);     
  }
}

void triggersend()  {
  digitalWrite(PULSEPIN, LOW);
  digitalWrite(PULSEPIN, HIGH);
  delay(2);
  digitalWrite(PULSEPIN, LOW);
}
