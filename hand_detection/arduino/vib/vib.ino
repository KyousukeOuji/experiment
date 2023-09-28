const int MOTOR_PIN = 6;
const int LED_PIN = 13;
const int BUFF_MAX = 30;
const int MOTOR_PINS[4] = {6, 9, 10, 11};

int counter;

void setup(){
    for (int i=0; i < 4; i++){
      pinMode(MOTOR_PINS[i], OUTPUT);
    }
    pinMode(LED_PIN, OUTPUT);
    counter = 0;
	Serial.begin(9600);
}

void loop(){

    String line;
    int line_len;
    int split_pos;
    String motor_str;
    String val_str;
    int motor_no;
    int val;

    line = Serial.readStringUntil('\n');
    line_len = line.length();

    if(line_len > 0){
            analogWrite(MOTOR_PINS[0], line[0]);
            analogWrite(MOTOR_PINS[1], line[1]);
            analogWrite(MOTOR_PINS[2], line[2]);
            analogWrite(MOTOR_PINS[3], line[3]);
    }
}
