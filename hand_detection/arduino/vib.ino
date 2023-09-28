const int MOTOR_PIN = 6;

void setup(){
	pinMode(MOTOR_PIN, OUTPUT);
	Serial.begin(9600);
}

byte inputData;
void loop(){
	if(Serial.available() > 0){
    inputData = Serial.parseInt();
    analogWrite(MOTOR_PIN, inputData);
	}
}
