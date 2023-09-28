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
        split_pos = line.indexOf(":");
        if (split_pos != -1){
            motor_str = line.substring(0, split_pos);
            val_str = line.substring(split_pos + 1);
            
            motor_no = motor_str.toInt();
            val = val_str.toInt();
            Serial.println(line);
            Serial.println(motor_str);
            analogWrite(MOTOR_PINS[motor_no], val);
        }
    }
}
