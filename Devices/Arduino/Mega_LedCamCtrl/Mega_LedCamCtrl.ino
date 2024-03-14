// constants
#define DEVICEINFO "Mega-Cam"
#define DEBUGPIN 12
#define NUMPINS 4   // allow maximum 4 pins to be controlled
#define TRIGGERPIN_REG INT0  // use pin 21 (PD0/SCL/INT0) to check the status of the cam to trigger external interrupt
#define TRIGGERPIN_STATUS_REG PIND  // port of the trigger pin
#define TRIGGERPIN 21
#define TRIGGERSENSE B01  // interrupt on any change (01 - any, 10 - failing edge, 11 - rising edge)
#define TRIGGERSENSE_REG ISC00  //register for the interrupt sensing control
#define OUTPUT_PORT PORTC
#define OUTPUT_PORT_CONF DDRC
#define OUTPUT_MASK B00001111   // first 4 pins in PORTC, which are 37 - 34 (?)

// parameters used to comunicate with computer
char mode = 'x';
char oldmode = 'x';
// char para[9];
byte b_para[9];   // parameters encoded with raw byte, so we can set parameters with 8 bit integers
int test = 0;

// define pins to use
// output pins to send trigger signal to LED and cam
// const trigger_pins[NUMPINS]

// pin state and delays
int ledState = HIGH;   // integer is 16 bit
long preMillis = 0;
long interval = 1000;
long preMicros = 0;

// cam control
// arduino receives the trigger out signal from the cam, and use it to control light source
// light triggers can be either all fired in each cycle, or alternating in different cycles
// select the mode with character 'c'
// first byte: which channels to use; 1 byte controls 8 digital channels
// second byte: if alternate 2nd-last channels; 0 for sequential alternating, 1 for not alternating; defaults to 0
int N_chs_inuse = 0;  // how many channels combinations are used
byte chs_list[NUMPINS * 2];  // channels to be activated with each ISR call
volatile int current_idx = 0;  // index of which channels to activate in next cycle
volatile byte next_out;        // channels to activate in next cycle
int byte_length = sizeof(b_para[0]) * 8;

// flags
bool device_running = false;


void setup() {
  // start serial port
  Serial.begin(19200);

  // setup led pin
  pinMode(DEBUGPIN, OUTPUT);

  // debug
  // digitalWrite(DEBUGPIN, HIGH);
  analogWrite(DEBUGPIN, 50);

  // test
  //Serial.write(TCCR0B);
  //Serial.write(TCCR0A);
}


void loop() {
  // read command from serial port and set mode accordingly
  if (Serial.available() > 10) {
    commandRead();
  }

  // different mode
  if (mode == 'i' | mode == 'I'){
    Serial.write(DEVICEINFO);
    mode = 'x';
    //Serial.println((int)b_para[0]);
    //Serial.write('\n');
  }

  // testing timing with millis and micros function
  if (mode == 'c' or mode == 'C'){
    LEDCtrl();
  }

  // used in Python to get baudrate
  if (mode == 'r' or mode == 'R'){
    Serial.write("brcheck");
  }
  
  // debug
  //delay(1000);
  //Serial.write(PIND);
}


// read serial command
void commandRead(){
    char bcheck= (char) Serial.read();
    if (bcheck=='b') {
      oldmode = mode;
      mode = (char) Serial.read();
      // for (int ii=0; ii<9; ii++) para[ii] = (char) Serial.read();
      for (int ii=0; ii<9; ii++) {
        b_para[ii] =  Serial.read();
      }
    }
}


// setting up the LED control
void setupCamLEDCtrl(){
  // setup trigger pin as input pin (value 0 is input), currently setting all pins on PORT to input
  pinMode(TRIGGERPIN, INPUT);
  // setup pin change interrupt, using external interrupt routine
  cli();
  EIMSK |= (1 << TRIGGERPIN_REG);   // external interrupt mask register, set INT0 to 1
  EICRA |= (TRIGGERSENSE << TRIGGERSENSE_REG);   // external interrupt control register A, 01 to detect any change

  // setup output pins as outputs
  OUTPUT_PORT_CONF = OUTPUT_MASK;
  // set all outputs to 0
  OUTPUT_PORT = 0;

  // disable the TC0 makes the interrupt handling much faster (jitter < 5 micro second)
  TCCR0B = 0;
  TCCR0A = 0;

  // enable global interrupts
  sei();

  // setup parameters based on command
  // which channels are used
  N_chs_inuse = 0;
  current_idx = 0;
  if (b_para[1] != 0){
    chs_list[0] = b_para[0];
    N_chs_inuse = 1;
  } else {
    for (int i=0; i < min(byte_length, NUMPINS); i++) {
      // check each bit to see if it is 1
      if (b_para[0] & (1 << i)){
        chs_list[N_chs_inuse] = 0b00000000 | (1 << i);
        N_chs_inuse++;
      }
    }
  }
  next_out = chs_list[0];
}


// stop camera LED control
void stopCamLEDCtrl(){
  cli();
  EIMSK ^= (1 << TRIGGERPIN_REG);  // stop interrupt
  OUTPUT_PORT = 0;  // turn all outputs to low
  OUTPUT_PORT_CONF = 0;  // turn all the output pins to inputs
  // enable TC0
  TCCR0B = 0b00000011;
  TCCR0A = 0b00000011;
  sei();
}


void LEDCtrl(){
  //toggle the led state in micro seconds circles
  device_running = true;
  setupCamLEDCtrl();
  while (device_running){
    if (Serial.available() > 10){
      commandRead();
      device_running = false;
      stopCamLEDCtrl();
    }
  }
}


// external change interrupt on INT0, with raw assembler code
// work to do: write next_out to the output port
ISR(INT0_vect, ISR_NAKED) {
  asm volatile (
    "    push r24                       \n"  // save content of r24
    "    lds r24,  %[port_val_adr]      \n"  // load content of next_out to r24
    "    out %[port],  r24              \n"  // write the output port output as r24
    "    pop  r24                       \n"  // restore old value of r24
    "    rjmp INT0_vect_part_2          \n"  // jump to part 2 of ISR
    :
    : [port_val_adr] "i" (&next_out),
      [port] "I" (_SFR_IO_ADDR(OUTPUT_PORT))
    : "r24"
  );
}


ISR(INT0_vect_part_2){
  // rising edge
  if (TRIGGERPIN_STATUS_REG & (1 << TRIGGERPIN_REG)) {
    // next digital out should be all off
    next_out = 0;
    // Serial.write("rising");
  }
    else {
    // set the idx to the next channel
    current_idx = (current_idx + 1) % N_chs_inuse;
    next_out = chs_list[current_idx];
  }
}

