#include <Wire.h>
#include <SD.h>
#include <SPI.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <sys/time.h>
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
//#include "MPU6050_6Axis_MotionApps612.h"
#include <sys/time.h>
#include <MAX30105.h>
#include "MAX30100.h"
#include "DFRobot_AHT20.h"

DFRobot_AHT20 aht20;
MAX30100 sensor;
MAX30105 particleSensor;
File dataFile;
File MPUdataFile;
File TEMPdataFile;
File BRdataFile;
char buffer[64];  // Bufor na pojedynczy wiersz danych
MPU6050 mpu;

float aht20temp = 0;

/*---MAX30102 Control Variables---*/
byte ledBrightness = 0x1F;
byte sampleAverage = 1;
byte ledMode = 2;
int sampleRate = 200;
int pulseWidth = 411;
int adcRange = 8192;

/*---MPU6050 Control/Status Variables---*/
bool DMPReady = false;   // Set true if DMP init was successful
uint8_t MPUIntStatus;    // Holds actual interrupt status byte from MPU
uint8_t devStatus;       // Return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;     // Expected DMP packet size (default is 42 bytes)
uint8_t FIFOBuffer[64];  // FIFO storage buffer
int XGyroOffset;
int YGyroOffset;
int ZGyroOffset;
int XAccelOffset;
int YAccelOffset;
int ZAccelOffset;

/*---Orientation/Motion Variables---*/
Quaternion q;         // [w, x, y, z]         Quaternion container
VectorInt16 aa;       // [x, y, z]            Accel sensor measurements
VectorInt16 gy;       // [x, y, z]            Gyro sensor measurements
VectorInt16 aaReal;   // [x, y, z]            Gravity-free accel sensor measurements
VectorInt16 aaWorld;  // [x, y, z]            World-frame accel sensor measurements
VectorFloat gravity;  // [x, y, z]            Gravity vector
float ypr[3];         // [yaw, pitch, roll]   Yaw/Pitch/Roll container and gravity vector
float axyz[3];        // [x, y, z]

#define SD_MOSI 18
#define SD_MISO 20
#define SD_SCLK 19
#define SD_CS 17

#define SAMPLE_DURATION_MS 120000
#define TIMER_INTERVAL_US 5000
#define MAX_SAMPLES 6000
#define CYCLE_DURATION_MS 300000
#define BUTTON_PIN_BITMASK ((1ULL << GPIO_NUM_1) | (1ULL << GPIO_NUM_2))
#define BR_SAMPLES 1500

const int MAX30102INT = 5;
const int BUTTON = 9;
const int SLEEP_INDICATOR_PIN = 21;  // Pin do wskazywania trybu sleep
const int MPU6050INT = 2;
const byte MAX30100INT = 4;

int activeBufferMAX = 1;
int cycleMAX = 0;
int ekgSamples1[MAX_SAMPLES];
int ekgSamples2[MAX_SAMPLES];
int deltaSamples1[MAX_SAMPLES];
long redSamples1[MAX_SAMPLES];
long irSamples1[MAX_SAMPLES];
int deltaSamples2[MAX_SAMPLES];
long redSamples2[MAX_SAMPLES];
long irSamples2[MAX_SAMPLES];
int sampleIndexMAX = 0;
int lastTime = 0;
volatile bool przycisk = false;
int fileIndex = 0;
float tempC;
const char *TEMPfilename = "/temperatura.csv";

bool sleepBR = 0;
int activeBufferBR = 1;
int cycleBR = 0;
int BRSamples1[BR_SAMPLES];
int BRSamples2[BR_SAMPLES];
int sampleIndexBR = 0;
int fileIndexBR = 0;

#define MAX_SAMPLES_MPU 1500
int activeBuffer = 1;
float yaw1[MAX_SAMPLES_MPU];
float pitch1[MAX_SAMPLES_MPU];
float roll1[MAX_SAMPLES_MPU];
int16_t ax1[MAX_SAMPLES_MPU];
int16_t ay1[MAX_SAMPLES_MPU];
int16_t az1[MAX_SAMPLES_MPU];
int16_t gx1[MAX_SAMPLES_MPU];
int16_t gy1[MAX_SAMPLES_MPU];
int16_t gz1[MAX_SAMPLES_MPU];
bool motion1[MAX_SAMPLES_MPU];
float yaw2[MAX_SAMPLES_MPU];
float pitch2[MAX_SAMPLES_MPU];
float roll2[MAX_SAMPLES_MPU];
int16_t ax2[MAX_SAMPLES_MPU];
int16_t ay2[MAX_SAMPLES_MPU];
int16_t az2[MAX_SAMPLES_MPU];
int16_t gx2[MAX_SAMPLES_MPU];
int16_t gy2[MAX_SAMPLES_MPU];
int16_t gz2[MAX_SAMPLES_MPU];
bool motion2[MAX_SAMPLES_MPU];
int sampleIndexMPU = 0;
int fileIndexMPU = 0;
unsigned long MPUCycleTime = 0;
int lasttimempu;
bool MPUSaving = false;
bool PPGSaving = false;
bool BRSaving = false;

uint32_t Vbatt = 0;
float Vbattf = 0;
bool temperatura = false;

enum State { IDLE_MAX,
             SAMPLING_MAX,
             SAVING_MAX,
             SLEEPING_MAX };
State currentStateMAX = IDLE_MAX;

enum State2 { IDLE_MPU,
              SAMPLING_MPU,
              SAVING_MPU };
State2 currentStateMPU = IDLE_MPU;

enum State3 { IDLE_BR,
              SAMPLING_BR,
              SAVING_BR,
              SLEEPING_BR };
State3 currentStateBR = IDLE_BR;

SemaphoreHandle_t MAXSavingSemaphore = NULL;
SemaphoreHandle_t MPUSavingSemaphore = NULL;
SemaphoreHandle_t BRSavingSemaphore = NULL;
SemaphoreHandle_t spiMutex;

/*------Interrupt detection routine------*/
volatile bool MPUInterrupt = false;  // Indicates whether MPU6050 interrupt pin has gone high
void IRAM_ATTR DMPDataReady() {
  MPUInterrupt = true;
  digitalWrite(SLEEP_INDICATOR_PIN, HIGH);
}

volatile bool sampleReadyMAX = false;
void IRAM_ATTR dataReadyMAX30102() {
  sampleReadyMAX = true;
  digitalWrite(SLEEP_INDICATOR_PIN, HIGH);  // Ustaw pin na LOW po wybudzeniu z trybu light sleep
}

volatile bool BRDataReady = false;
void BRInterrupt() {
  BRDataReady = true;
}

void IRAM_ATTR Przycisk() {
  przycisk = true;
}

String generateFileName(int index) {
  return "/PPG" + String(index) + ".csv";
}

String generateFileNameMPU(int index) {
  return "/IMU" + String(index) + ".csv";
}

String generateFileNameBR(int index) {
  return "/BR" + String(index) + ".csv";
}

void changeMux(uint8_t bus){
  Wire.beginTransmission(0x70);
  Wire.write(1 << bus);
  Wire.endTransmission();
}

void setSystemTimeFromFile() {
  // Otwórz plik z czasem na karcie SD
  File timeFile = SD.open("/time.txt");
  if (!timeFile) {
    Serial.println("Nie udało się otworzyć pliku z czasem.");
    return;
  }

  // Odczytanie daty i godziny z pliku
  String timeStr = timeFile.readStringUntil('\n');  // Odczytanie linii z pliku
  timeFile.close();

  int day, month, year, hour, minute, second;
  if (sscanf(timeStr.c_str(), "%d/%d/%d %d:%d:%d", &day, &month, &year, &hour, &minute, &second) == 6) {
    // Konwersja na format systemowy
    struct tm timeinfo;
    timeinfo.tm_year = year - 1900;  // Rok zaczyna się od 1900
    timeinfo.tm_mon = month - 1;     // Miesiące zaczynają się od 0
    timeinfo.tm_mday = day;
    timeinfo.tm_hour = hour;
    timeinfo.tm_min = minute;
    timeinfo.tm_sec = second;

    time_t t = mktime(&timeinfo);  // Przekształcamy na time_t
    struct timeval tv = { t, 0 };  // Ustawiamy czas
    settimeofday(&tv, NULL);

    Serial.println("Czas ustawiony na podstawie pliku.");
  } else {
    Serial.println("Błąd formatu daty w pliku.");
  }
}

void enterLightSleep() {
  digitalWrite(SLEEP_INDICATOR_PIN, LOW);  // Ustaw pin na HIGH przed wejściem w tryb light sleep
  esp_sleep_enable_ext1_wakeup(BUTTON_PIN_BITMASK, ESP_EXT1_WAKEUP_ANY_LOW);
  esp_light_sleep_start();
}

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("Inicjalizacja...");

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  pinMode(MAX30100INT, INPUT_PULLUP);
  pinMode(MPU6050INT, INPUT);
  pinMode(MAX30102INT, INPUT_PULLUP);
  pinMode(BUTTON, INPUT_PULLUP);
  pinMode(SLEEP_INDICATOR_PIN, OUTPUT);  // Ustawienie pinu do wskazywania trybu sleep
  digitalWrite(SLEEP_INDICATOR_PIN, LOW);
  analogReadResolution(12);

  Vbatt = analogReadMilliVolts(0); // Read and accumulate ADC voltage
  Vbattf = 3 * Vbatt/ 1000.0;
  Vbatt = 0;

  attachInterrupt(digitalPinToInterrupt(MAX30102INT), dataReadyMAX30102, FALLING);
  attachInterrupt(digitalPinToInterrupt(BUTTON), Przycisk, FALLING);
  attachInterrupt(digitalPinToInterrupt(MAX30100INT), BRInterrupt, FALLING);

  SPI.begin(SD_SCLK, SD_MISO, SD_MOSI, SD_CS);
  delay(5);
  if (!SD.begin(SD_CS, SPI, 80000000)) {
    Serial.println("Nie udało się zamontować karty SD");
    while (1) {
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(1700);
    }
  }

  Serial.println("Karta SD wykryta");

  // Ustawienie czasu na podstawie pliku
  setSystemTimeFromFile();

  File file = SD.open("/SensorSetup.txt");
  if (file) {
    while (file.available()) {
      String line = file.readStringUntil('\n');
      line.trim();
      if (line.startsWith("ledBrightness=")) {
        ledBrightness = line.substring(14).toInt();
      } else if (line.startsWith("sampleAverage=")) {
        sampleAverage = line.substring(14).toInt();
      } else if (line.startsWith("ledMode=")) {
        ledMode = line.substring(8).toInt();
      } else if (line.startsWith("sampleRate=")) {
        sampleRate = line.substring(11).toInt();
      } else if (line.startsWith("pulseWidth=")) {
        pulseWidth = line.substring(11).toInt();
      } else if (line.startsWith("adcRange=")) {
        adcRange = line.substring(9).toInt();
      } else if (line.startsWith("XGyroOffset=")) {
        XGyroOffset = line.substring(12).toInt();
      } else if (line.startsWith("YGyroOffset=")) {
        YGyroOffset = line.substring(12).toInt();
      } else if (line.startsWith("ZGyroOffset=")) {
        ZGyroOffset = line.substring(12).toInt();
      } else if (line.startsWith("XAccelOffset=")) {
        XAccelOffset = line.substring(13).toInt();
      } else if (line.startsWith("YAccelOffset=")) {
        YAccelOffset = line.substring(13).toInt();
      } else if (line.startsWith("ZAccelOffset=")) {
        ZAccelOffset = line.substring(13).toInt();
      }
    }
    file.close();
  }

  Wire.begin();
  Wire.setClock(400000);
  changeMux(1);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Nie znaleziono MAX30102.");
    while (1) {
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(1100);
    }
  } else {
    Serial.println("MAX30102 - zainicjowany");
  }

  particleSensor.softReset();
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  // particleSensor.enableFIFORollover();
  particleSensor.enableDATARDY();
  Serial.println("MAX30102 - przerwania włączone");
  /*---------------------------------------------------------------------------*/

  /*Initialize device*/
  Serial.println(F("Initializing IMU device..."));
  mpu.initialize();
  /*Verify connection*/
  Serial.println(F("Testing MPU6050 connection..."));
  if (mpu.testConnection() == false) {
    Serial.println("MPU6050 connection failed");
    while (true) {
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(1500);
    };
  } else {
    Serial.println("MPU6050 connection successful");
  }
  /* Initializate and configure the DMP*/
  Serial.println(F("Initializing DMP..."));
  devStatus = mpu.dmpInitialize();

  /* Supply your gyro offsets here, scaled for min sensitivity */
  mpu.setXGyroOffset(XGyroOffset);
  mpu.setYGyroOffset(YGyroOffset);
  mpu.setZGyroOffset(ZGyroOffset);
  mpu.setXAccelOffset(XAccelOffset);
  mpu.setYAccelOffset(YAccelOffset);
  mpu.setZAccelOffset(ZAccelOffset);

  /* Making sure it worked (returns 0 if so) */
  if (devStatus == 0) {
    // delay(10000);
    // mpu.CalibrateAccel(10);  // Calibration Time: generate offsets and calibrate our MPU6050
    // mpu.CalibrateGyro(10);
    // Serial.println("These are the Active offsets: ");
    // mpu.PrintActiveOffsets();
    mpu.setInterruptLatch(true);
    mpu.setInterruptLatchClear(true);
    mpu.setInterruptMode(true);
    mpu.setZeroMotionDetectionDuration(3);
    mpu.setZeroMotionDetectionThreshold(1);
    Serial.println(F("Enabling DMP..."));  //Turning ON DMP
    mpu.setDMPEnabled(true);

    /*Enable Arduino interrupt detection*/
    Serial.print(F("Enabling interrupt detection (Arduino external interrupt "));
    Serial.print(digitalPinToInterrupt(MPU6050INT));
    Serial.println(F(")..."));
    attachInterrupt(digitalPinToInterrupt(MPU6050INT), DMPDataReady, FALLING);
    Serial.println(mpu.get_acce_resolution());
    Serial.println(mpu.get_gyro_resolution());
    Serial.println(mpu.getRate());
    Serial.println(mpu.getFullScaleGyroRange());
    Serial.println(mpu.getFullScaleAccelRange());
    MPUIntStatus = mpu.getIntStatus();

    /* Set the DMP Ready flag so the main loop() function knows it is okay to use it */
    Serial.println(F("DMP ready! Waiting for first interrupt..."));
    DMPReady = true;
    packetSize = mpu.dmpGetFIFOPacketSize();  //Get expected DMP packet size for later comparison
  } else {
    Serial.print(F("DMP Initialization failed (code "));  //Print the error code
    Serial.print(devStatus);
    Serial.println(F(")"));
    while (true) {
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(900);
    }
    // 1 = initial memory load failed
    // 2 = DMP configuration updates failed
  }

  /*---------------------------------------------------------------------------*/
  changeMux(0);
  sensor.begin(pw400, i21, sr50);
  sensor.setInterrupt(hr);

  changeMux(1);
  particleSensor.getINT1();

  uint8_t status;
  // while((status = aht20.begin()) != 0){
  //     digitalWrite(LED_BUILTIN, LOW);
  //     delay(300);
  //     digitalWrite(LED_BUILTIN, HIGH);
  //     delay(300);
  //     digitalWrite(LED_BUILTIN, LOW);
  //     delay(300);
  //     digitalWrite(LED_BUILTIN, HIGH);
  //     delay(300);
  //     digitalWrite(LED_BUILTIN, LOW);
  //     delay(300);
  //     digitalWrite(LED_BUILTIN, HIGH);
  //     delay(300);
  //     digitalWrite(LED_BUILTIN, LOW);
  //     delay(300);
  //     digitalWrite(LED_BUILTIN, HIGH);
  //     delay(300);
  //     digitalWrite(LED_BUILTIN, LOW);
  //     delay(300);
  //     digitalWrite(LED_BUILTIN, HIGH);
  //     delay(900);
  // }

  MPUSavingSemaphore = xSemaphoreCreateBinary();
  MAXSavingSemaphore = xSemaphoreCreateBinary();
  BRSavingSemaphore = xSemaphoreCreateBinary();
  spiMutex = xSemaphoreCreateMutex();

  if (MAXSavingSemaphore == NULL) {
    Serial.println("Nie udało się utworzyć semafora MAX");
    while (1) {
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(900);
    }
  }

  if (MPUSavingSemaphore == NULL) {
    Serial.println("Nie udało się utworzyć semafora MPU");
    while (1) {
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(900);
    }
  }

  if (spiMutex == NULL) {
    Serial.println("Nie udało się utworzyć mutexu SPI");
    while (1) {
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(900);
    }
  }
  digitalWrite(LED_BUILTIN, LOW);
  xTaskCreatePinnedToCore(taskSaveData, "TaskSaveData", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(taskSaveDataMPU, "TaskSaveDataMPU", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(taskSaveDataBR, "TaskSaveDataBR", 4096, NULL, 1, NULL, 0);
}

void loop() {
  static unsigned long totalCycleTime = millis();

  switch (currentStateMAX) {
    case IDLE_MAX:
      {
        if (sampleReadyMAX) {
          sampleReadyMAX = false;
          currentStateMAX = SAMPLING_MAX;
        }
        break;
      }
    case SAMPLING_MAX:
      {
        changeMux(1);
        if (activeBufferMAX == 1) {
          particleSensor.check();
          deltaSamples1[sampleIndexMAX] = millis() - lastTime;
          irSamples1[sampleIndexMAX] = particleSensor.getFIFOIR();
          redSamples1[sampleIndexMAX] = particleSensor.getFIFORed();
          // ekgSamples1[sampleIndexMAX] = analogRead(1);
        } else {
          particleSensor.check();
          deltaSamples2[sampleIndexMAX] = millis() - lastTime;
          irSamples2[sampleIndexMAX] = particleSensor.getFIFOIR();
          redSamples2[sampleIndexMAX] = particleSensor.getFIFORed();
          // ekgSamples2[sampleIndexMAX] = analogRead(1);
        }
        sampleIndexMAX++;
        lastTime = millis();
        particleSensor.nextSample();
        if (sampleIndexMAX >= MAX_SAMPLES) {
          if (activeBufferMAX == 1) {
            activeBufferMAX = 2;
          } else {
            activeBufferMAX = 1;
          }
          sampleIndexMAX = 0;
          cycleMAX++;
          if (cycleMAX >= 4) {
            changeMux(1);
            digitalWrite(LED_BUILTIN, HIGH);
            particleSensor.shutDown();
            currentStateMAX = SLEEPING_MAX;
            temperatura = true;
          }else{
            currentStateMAX = IDLE_MAX;
          }
          xSemaphoreGive(MAXSavingSemaphore);  // Daj sygnał semaforowi
        } else {
          currentStateMAX = IDLE_MAX;
          //enterLightSleep();
        }
        break;
      }
    case SAVING_MAX:
      {
        // Task do zapisu danych zajmie się zapisem
        break;
      }
    case SLEEPING_MAX:
      {
        //Serial.println("Stan SLEEPING_MAX");
        if ((millis() - totalCycleTime) >= CYCLE_DURATION_MS) {
          // if(aht20.startMeasurementReady(/* crcEn = */true)){
          //   aht20temp = aht20.getTemperature_C();
          // }
          Vbatt = analogReadMilliVolts(0); // Read and accumulate ADC voltage
          Vbattf = 3 * Vbatt/ 1000.0;
          Vbatt = 0;
          changeMux(1);
          cycleMAX = 0;
          currentStateMAX = IDLE_MAX;
          particleSensor.wakeUp();
          particleSensor.enableDIETEMPRDY();
          tempC = particleSensor.readTemperature();
          particleSensor.disableDIETEMPRDY();
          particleSensor.getINT1();
          particleSensor.getINT2();
          particleSensor.clearFIFO();
          totalCycleTime = millis();
          sleepBR = 0;
          digitalWrite(LED_BUILTIN, LOW);
          Serial.println("Cykl resetowany");
        }
        break;
      }
  }

  switch (currentStateMPU) {
    case IDLE_MPU:
      {
        if (MPUInterrupt) {
          MPUInterrupt = false;
          currentStateMPU = SAMPLING_MPU;
        }
        if (przycisk) {
          przycisk = false;
          Serial.println(particleSensor.getINT1());
          Serial.println("Nic nie robię");
        }
        break;
      }
    case SAMPLING_MPU:
      {
        mpu.dmpGetCurrentFIFOPacket(FIFOBuffer);
        /* Display Euler angles in degrees */
        mpu.dmpGetQuaternion(&q, FIFOBuffer);
        mpu.dmpGetAccel(&aa, FIFOBuffer);
        mpu.dmpGetGyro(&gy, FIFOBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
        //mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
        //mpu.dmpConvertToWorldFrame(&aaWorld, &aaReal, &q);
        if (activeBuffer == 1) {
          yaw1[sampleIndexMPU] = ypr[0] * 180 / M_PI;
          pitch1[sampleIndexMPU] = ypr[1] * 180 / M_PI;
          roll1[sampleIndexMPU] = ypr[2] * 180 / M_PI;
          ax1[sampleIndexMPU] = aa.x;
          ay1[sampleIndexMPU] = aa.y;
          az1[sampleIndexMPU] = aa.z;
          gx1[sampleIndexMPU] = gy.x;
          gy1[sampleIndexMPU] = gy.y;
          gz1[sampleIndexMPU] = gy.z;
          // motion1[sampleIndexMPU] = mpu.getZeroMotionDetected();
        } else {
          yaw2[sampleIndexMPU] = ypr[0] * 180 / M_PI;
          pitch2[sampleIndexMPU] = ypr[1] * 180 / M_PI;
          roll2[sampleIndexMPU] = ypr[2] * 180 / M_PI;
          ax2[sampleIndexMPU] = aa.x;
          ay2[sampleIndexMPU] = aa.y;
          az2[sampleIndexMPU] = aa.z;
          gx2[sampleIndexMPU] = gy.x;
          gy2[sampleIndexMPU] = gy.y;
          gz2[sampleIndexMPU] = gy.z;
          // motion2[sampleIndexMPU] = mpu.getZeroMotionDetected();
        }
        sampleIndexMPU++;
        lasttimempu = millis();
        if (sampleIndexMPU >= MAX_SAMPLES_MPU) {
          currentStateMPU = IDLE_MPU;
          if (activeBuffer == 1) {
            activeBuffer = 2;
          } else {
            activeBuffer = 1;
          }
          sampleIndexMPU = 0;
          xSemaphoreGive(MPUSavingSemaphore);  // Daj sygnał semaforowi
        } else {
          //enterLightSleep();
          currentStateMPU = IDLE_MPU;
        }
        break;
      }
    case SAVING_MPU:
      {
        // Task do zapisu danych zajmie się zapisem
        break;
      }
  }

  switch (currentStateBR) {
    case IDLE_BR:
      {
        if (BRDataReady) {
          BRDataReady = false;
          currentStateBR = SAMPLING_BR;
        }
        break;
      }
    case SAMPLING_BR:
      {
        changeMux(0);
        if (activeBufferBR == 1) {
          sensor.readSensor();
          BRSamples1[sampleIndexBR] = sensor.IR;

        } else {
          sensor.readSensor();
          BRSamples2[sampleIndexBR] = sensor.IR;

        }
        sampleIndexBR++;
        if (sampleIndexBR >= BR_SAMPLES) {
          if (activeBufferBR == 1) {
            activeBufferBR = 2;
          } else {
            activeBufferBR = 1;
          }
          sampleIndexBR = 0;
          cycleBR++;
          if (cycleBR >= 4) {
            changeMux(0);
            sensor.shutdown();
            sampleIndexBR = 0;
            sleepBR = 1;
            currentStateBR = SLEEPING_BR;
          }else{
            currentStateBR = IDLE_BR;
          }
          xSemaphoreGive(BRSavingSemaphore);  // Daj sygnał semaforowi
        } else {
          currentStateBR = IDLE_BR;
        }
        break;
      }
    case SAVING_BR:
      {
        // Task do zapisu danych zajmie się zapisem
        break;
      }
    case SLEEPING_BR:
      {
        if (sleepBR == 0) {
          changeMux(0);
          currentStateBR= IDLE_BR;
          cycleBR = 0;
          sensor.startup();
          digitalWrite(LED_BUILTIN, LOW);
          Serial.println("Cykl resetowany");
        }
        break;
      }
  }

  if (currentStateMAX == SLEEPING_MAX && currentStateMPU == IDLE_MPU && currentStateBR == SLEEPING_BR) {
    if(millis()-lasttimempu <= 10 && MPUSaving == false && PPGSaving == false && BRSaving == false){
      enterLightSleep();
    }
  }
}

void taskSaveData(void *pvParameters) {
  while (true) {
    if (xSemaphoreTake(MAXSavingSemaphore, portMAX_DELAY) == pdTRUE) {
      xSemaphoreTake(spiMutex, portMAX_DELAY);
      PPGSaving = true;
      Serial.println("Stan SAVING_MAX");
      Serial.println("Zapisuję dane");
      String fileName = generateFileName(fileIndex++);
      unsigned long savingCycleTime1 = millis();
      dataFile = SD.open(fileName.c_str(), FILE_WRITE);
      if (dataFile) {
        dataFile.println("RED,IR,TIME");
        if (activeBufferMAX == 2) {
          for (int i = 0; i < MAX_SAMPLES; i++) {
            dataFile.print(redSamples1[i]);
            dataFile.print(",");
            dataFile.print(irSamples1[i]);
            dataFile.print(",");
            // dataFile.print(ekgSamples1[i]);
            // dataFile.print(",");
            dataFile.println(deltaSamples1[i]);
          }
        } else {
          for (int i = 0; i < MAX_SAMPLES; i++) {
            dataFile.print(redSamples2[i]);
            dataFile.print(",");
            dataFile.print(irSamples2[i]);
            dataFile.print(",");
            // dataFile.print(ekgSamples2[i]);
            // dataFile.print(",");
            dataFile.println(deltaSamples2[i]);
          }
        }
        dataFile.close();
        Serial.print("Czas zapisu: ");
        Serial.println(millis() - savingCycleTime1);
        Serial.print("Dane zapisane w pliku: ");
        Serial.println(fileName);
      } else {
        Serial.println("Błąd podczas zapisywania danych");
      }
      if (temperatura) {
        temperatura = false;
        sampleIndexMAX = 0;
        if (fileIndex == 1) {
          TEMPdataFile = SD.open(TEMPfilename, FILE_WRITE);
          if (TEMPdataFile) {
            // TEMPdataFile.println("Temperatura MAX [C], Temperatura AHT20 [C]");  // Nagłówek pliku
            TEMPdataFile.println("Temperatura MAX [C], Napięcie baterii [V]");  // Nagłówek pliku
          }
        } else {
          TEMPdataFile = SD.open(TEMPfilename, FILE_APPEND);
        }

        if (TEMPdataFile) {
          TEMPdataFile.print(tempC, 2);  // Zapis temperatury z dwoma miejscami po przecinku
          TEMPdataFile.print(",");
          TEMPdataFile.println(Vbattf);
          // TEMPdataFile.println(aht20temp, 2);  // Zapis temperatury z dwoma miejscami po przecinku
          TEMPdataFile.close();
          Serial.println("Dane zapisane.");
        } else {
          Serial.println("Błąd otwierania pliku!");
        }
      }
      PPGSaving = false;
      xSemaphoreGive(spiMutex);
    }
  }
}

void taskSaveDataMPU(void *pvParameters) {
  while (true) {
    if (xSemaphoreTake(MPUSavingSemaphore, portMAX_DELAY) == pdTRUE) {
      xSemaphoreTake(spiMutex, portMAX_DELAY);
      MPUSaving = true;
      Serial.println("Stan SAVING_MPU");
      Serial.println("Zapisuję dane MPU");
      Serial.print("Czas cyklu MPU: ");
      Serial.println(millis() - MPUCycleTime);
      MPUCycleTime = millis();
      unsigned long savingCycleTime2 = millis();
      String MPUfileName = generateFileNameMPU(fileIndexMPU++);
      MPUdataFile = SD.open(MPUfileName.c_str(), FILE_WRITE);
      if (MPUdataFile) {
        MPUdataFile.println("Yaw,Pitch,Roll,AX,AY,AZ,GX,GY,GZ");
        if (activeBuffer == 2) {
          for (int i = 0; i < MAX_SAMPLES_MPU; i++) {
            MPUdataFile.print(yaw1[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(pitch1[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(roll1[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(ax1[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(ay1[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(az1[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(gx1[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(gy1[i]);
            MPUdataFile.print(",");
            MPUdataFile.println(gz1[i]);
          }
        } else {
          for (int i = 0; i < MAX_SAMPLES_MPU; i++) {
            MPUdataFile.print(yaw2[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(pitch2[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(roll2[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(ax2[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(ay2[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(az2[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(gx2[i]);
            MPUdataFile.print(",");
            MPUdataFile.print(gy2[i]);
            MPUdataFile.print(",");
            MPUdataFile.println(gz2[i]);
          }
        }
        MPUdataFile.close();
      } else {
        Serial.println("Błąd podczas zapisywania danych");
      }
      Serial.print("Czas zapisu MPU: ");
      Serial.println(millis() - savingCycleTime2);
      MPUSaving = false;
      xSemaphoreGive(spiMutex);
    }
  }
}

void taskSaveDataBR(void *pvParameters) {
  while (true) {
    if (xSemaphoreTake(BRSavingSemaphore, portMAX_DELAY) == pdTRUE) {
      xSemaphoreTake(spiMutex, portMAX_DELAY);
      BRSaving = true;
      Serial.println("Stan SAVING_BR");
      Serial.println("Zapisuję BR");
      String fileNameBR = generateFileNameBR(fileIndexBR++);
      BRdataFile = SD.open(fileNameBR.c_str(), FILE_WRITE);
      if (BRdataFile) {
        BRdataFile.println("BR");
        if (activeBufferBR == 2) {
          for (int i = 0; i < BR_SAMPLES; i++) {
            BRdataFile.println(BRSamples1[i]);
          }
        } else {
          for (int i = 0; i < BR_SAMPLES; i++) {
            BRdataFile.println(BRSamples2[i]);
          }
        }
        BRdataFile.close();
        Serial.print("Dane zapisane w pliku: ");
        Serial.println(fileNameBR);
      } else {
        Serial.println("Błąd podczas zapisywania danych");
      }
      BRSaving = false;
      xSemaphoreGive(spiMutex);
    }
  }
}