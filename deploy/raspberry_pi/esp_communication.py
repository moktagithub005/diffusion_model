"""
ESP Communication Module

This module handles communication with ESP32/ESP8266 devices
for controlling motors and other actuators in response to hail detection.
"""

import requests
import logging
import socket
import time
import json

logger = logging.getLogger('hail_detection.esp')

class ESPCommunicator:
    """
    Handles communication with ESP module for hail protection system.
    
    Supports both HTTP and Socket-based communication.
    """
    def __init__(self, ip_address, port, protocol="http", timeout=5):
        """
        Initialize the ESP communicator.
        
        Args:
            ip_address: IP address of the ESP module
            port: Port number for communication
            protocol: Communication protocol ("http" or "socket")
            timeout: Connection timeout in seconds
        """
        self.ip_address = ip_address
        self.port = port
        self.protocol = protocol.lower()
        self.timeout = timeout
        
        # Validate protocol
        if self.protocol not in ["http", "socket"]:
            raise ValueError(f"Unsupported protocol: {protocol}. Use 'http' or 'socket'.")
        
        logger.info(f"Initialized ESP communicator with {protocol} protocol to {ip_address}:{port}")
    
    def send_command(self, command, params=None):
        """
        Send a command to the ESP module.
        
        Args:
            command: Command string ("activate", "deactivate", "status", etc.)
            params: Optional parameters dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if self.protocol == "http":
            return self._send_http_command(command, params)
        else:  # socket
            return self._send_socket_command(command, params)
    
    def _send_http_command(self, command, params=None):
        """Send command using HTTP."""
        try:
            # Prepare URL and payload
            url = f"http://{self.ip_address}:{self.port}/{command}"
            
            # Make request
            if params:
                response = requests.post(url, json=params, timeout=self.timeout)
            else:
                response = requests.get(url, timeout=self.timeout)
            
            # Check response
            response.raise_for_status()
            
            # Log response
            logger.info(f"ESP command '{command}' successful: {response.text}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending HTTP command to ESP: {e}")
            return False
    
    def _send_socket_command(self, command, params=None):
        """Send command using socket connection."""
        try:
            # Create socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Set timeout
                s.settimeout(self.timeout)
                
                # Connect
                s.connect((self.ip_address, self.port))
                
                # Prepare message
                message = {
                    "command": command
                }
                
                if params:
                    message["params"] = params
                
                # Send message
                s.sendall(json.dumps(message).encode('utf-8'))
                
                # Get response
                response = s.recv(1024).decode('utf-8')
                
                # Log response
                logger.info(f"ESP command '{command}' response: {response}")
                
                # Check for success
                try:
                    response_data = json.loads(response)
                    return response_data.get("success", False)
                except (json.JSONDecodeError, KeyError):
                    logger.error(f"Invalid response from ESP: {response}")
                    return False
                
        except (socket.error, socket.timeout) as e:
            logger.error(f"Socket error communicating with ESP: {e}")
            return False
    
    def check_status(self):
        """
        Check the status of the ESP module.
        
        Returns:
            Status dictionary or None if failed
        """
        try:
            if self.protocol == "http":
                url = f"http://{self.ip_address}:{self.port}/status"
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            else:  # socket
                return self._send_socket_command("status")
        except Exception as e:
            logger.error(f"Error checking ESP status: {e}")
            return None
    
    def activate_protection(self, duration_minutes=None):
        """
        Activate hail protection.
        
        Args:
            duration_minutes: Optional duration in minutes
            
        Returns:
            True if successful, False otherwise
        """
        params = {}
        if duration_minutes is not None:
            params["duration"] = duration_minutes
        
        return self.send_command("activate", params)
    
    def deactivate_protection(self):
        """
        Deactivate hail protection.
        
        Returns:
            True if successful, False otherwise
        """
        return self.send_command("deactivate")

# Sample ESP32/ESP8266 Arduino code for reference
"""
ESP32/ESP8266 Arduino code for the hail protection system

#include <WiFi.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "YourWiFiSSID";
const char* password = "YourWiFiPassword";

// Motor control pins
const int motorPin1 = 5;  // D1 on ESP8266, GPIO5 on ESP32
const int motorPin2 = 4;  // D2 on ESP8266, GPIO4 on ESP32
const int motorEnablePin = 2;  // D4 on ESP8266, GPIO2 on ESP32

// Server setup
WiFiServer server(8080);
bool protectionActive = false;

void setup() {
  Serial.begin(115200);
  
  // Set up motor control pins
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(motorEnablePin, OUTPUT);
  
  // Stop motor initially
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);
  digitalWrite(motorEnablePin, LOW);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  
  // Start the server
  server.begin();
}

void loop() {
  // Check if a client has connected
  WiFiClient client = server.available();
  if (!client) {
    return;
  }
  
  // Wait for data
  while (client.connected() && !client.available()) {
    delay(1);
  }
  
  // Read the first line of the request
  String request = client.readStringUntil('\r');
  client.flush();
  
  // Parse JSON message
  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, request);
  
  // Prepare response
  DynamicJsonDocument response(1024);
  
  if (error) {
    response["success"] = false;
    response["error"] = "Invalid JSON";
  } else {
    String command = doc["command"];
    
    if (command == "activate") {
      // Activate protection
      activateMotor();
      protectionActive = true;
      response["success"] = true;
      response["message"] = "Protection activated";
    } 
    else if (command == "deactivate") {
      // Deactivate protection
      deactivateMotor();
      protectionActive = false;
      response["success"] = true;
      response["message"] = "Protection deactivated";
    }
    else if (command == "status") {
      response["success"] = true;
      response["active"] = protectionActive;
    }
    else {
      response["success"] = false;
      response["error"] = "Unknown command";
    }
  }
  
  // Send response
  String responseStr;
  serializeJson(response, responseStr);
  client.println(responseStr);
  
  delay(1);
  client.stop();
}

void activateMotor() {
  // Deploy the nets by running the motor
  digitalWrite(motorPin1, HIGH);
  digitalWrite(motorPin2, LOW);
  digitalWrite(motorEnablePin, HIGH);
  
  // Run motor for 10 seconds to deploy
  delay(10000);
  
  // Stop motor
  digitalWrite(motorEnablePin, LOW);
}

void deactivateMotor() {
  // Retract the nets by running the motor in reverse
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, HIGH);
  digitalWrite(motorEnablePin, HIGH);
  
  // Run motor for 10 seconds to retract
  delay(10000);
  
  // Stop motor
  digitalWrite(motorEnablePin, LOW);
}
"""