// Experimento 2 - Aula 1 - SENSOR DE ULTRASSOM COM TRES DISTANCIAS

//Conexão com sensor de ultrassom HC - SR04

//CARREGANDO BIBLIOTECA
#include <Ultrasonic.h>

//DEFININDO OS PINOS PARA TRIGGER E ECHO
#define pino_trigger 4
#define pino_echo 5

//INICIALIZANDO O SENSOR NOS PINOS PRE DEFINIDOS
Ultrasonic ultrasonic(pino_trigger, pino_echo);

//DECLARACAO DOS LEDS
int LEDverde = 10;
int LEDvermelho = 11;
int LEDvermelho = 12;

void setup()
{
  //MODULO HC-04
  Serial.begin(9600);
  Serial.println("Lendo dados do sensor..");

  //LED
  pinMode(LEDverde,OUTPUT);
  pinMode(LEDvermelho,OUTPUT);
  pinMode(LEDamarelo,OUTPUT);
}

void loop()
{
  //LENDO AS INFORMACOES DO SENSOR EM CM E POL.
  float cmMsec;
  long microsec = ultrasonic.timing();
  cmMsec = ultrasonic.convert(microsec, Ultrasonic::CM);

  //SE DISTANCIA >16 LED VERDE, SE <10 E >5 LED AMARELO, SE <5 LED VERMELHO
  if (cmMsec > 16){
    digitalWrite(LEDverde,HIGH);
    digitalWrite(LEDvermelho,LOW);
    digitalWrite(LEDamarelo,LOW);
  } else if (cmSec < 16 && cmMsec >12) { 
    digitalWrite(LEDverde,HIGH);
    digitalWrite(LEDvermelho,LOW);
    digitalWrite(LEDamarelo,LOW);  }
  } else if (cmSec < 12) {
    digitalWrite(LEDverde,LOW);
    digitalWrite(LEDvermelho,LOW);
    digitalWrite(LEDamarelo,HIGH);

//EXIBINDO INFORMACOES NO SERIAL MONITOR
  Serial.print("Distancia em cm:");
  Serial.println(cmMsec);
  
//PAUSA PARA ENVIAR MENSAGENS
delay(500);
}
