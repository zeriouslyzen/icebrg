"""
Mobile Templates for ICEBURG
Generates React Native and Android Studio projects for mobile app development.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MobilePlatform(Enum):
    """Mobile platform types."""
    REACT_NATIVE = "react_native"
    ANDROID_STUDIO = "android_studio"
    FLUTTER = "flutter"


@dataclass
class MobileProject:
    """Mobile project structure."""
    project_name: str
    project_path: str
    platform: MobilePlatform
    features: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MobileTemplate:
    """Mobile template generator."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.template_dir = self.config.get("template_dir", "templates/mobile")
        os.makedirs(self.template_dir, exist_ok=True)
    
    async def generate_react_native_project(self, 
                                          project_name: str,
                                          description: str,
                                          features: List[str] = None) -> MobileProject:
        """Generate React Native project."""
        project_path = os.path.join(self.template_dir, project_name)
        os.makedirs(project_path, exist_ok=True)
        
        # Create React Native project structure
        await self._create_react_native_structure(project_path)
        
        # Generate package.json
        package_json = {
            "name": project_name.lower().replace(" ", "-"),
            "version": "1.0.0",
            "description": description,
            "main": "index.js",
            "scripts": {
                "start": "react-native start",
                "android": "react-native run-android",
                "ios": "react-native run-ios"
            },
            "dependencies": {
                "react": "18.2.0",
                "react-native": "0.72.0",
                "@react-navigation/native": "^6.1.0",
                "@react-navigation/stack": "^6.3.0",
                "react-native-screens": "^3.25.0",
                "react-native-safe-area-context": "^4.7.0"
            }
        }
        
        with open(os.path.join(project_path, "package.json"), "w") as f:
            json.dump(package_json, f, indent=2)
        
        # Generate App.js
        app_js = f"""import React from 'react';
import {{ NavigationContainer }} from '@react-navigation/native';
import {{ createStackNavigator }} from '@react-navigation/stack';
import HomeScreen from './src/screens/HomeScreen';
import DetailScreen from './src/screens/DetailScreen';

const Stack = createStackNavigator();

export default function App() {{
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={{HomeScreen}} />
        <Stack.Screen name="Detail" component={{DetailScreen}} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}}
"""
        
        with open(os.path.join(project_path, "App.js"), "w") as f:
            f.write(app_js)
        
        # Generate screens
        await self._generate_react_native_screens(project_path)
        
        return MobileProject(
            project_name=project_name,
            project_path=project_path,
            platform=MobilePlatform.REACT_NATIVE,
            features=features or [],
            metadata={"description": description}
        )
    
    async def generate_android_studio_project(self, 
                                            project_name: str,
                                            description: str,
                                            features: List[str] = None) -> MobileProject:
        """Generate Android Studio project."""
        project_path = os.path.join(self.template_dir, project_name)
        os.makedirs(project_path, exist_ok=True)
        
        # Create Android project structure
        await self._create_android_structure(project_path)
        
        # Generate build.gradle
        build_gradle = f"""plugins {{
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}}

android {{
    namespace 'com.iceburg.{project_name.lower()}'
    compileSdk 34

    defaultConfig {{
        applicationId "com.iceburg.{project_name.lower()}"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "1.0"
    }}

    buildTypes {{
        release {{
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }}
    }}
}}

dependencies {{
    implementation 'androidx.core:core-ktx:1.10.1'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.9.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'
}}
"""
        
        with open(os.path.join(project_path, "app/build.gradle"), "w") as f:
            f.write(build_gradle)
        
        # Generate MainActivity.kt
        main_activity = f"""package com.iceburg.{project_name.lower()}

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {{
    override fun onCreate(savedInstanceState: Bundle?) {{
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }}
}}
"""
        
        with open(os.path.join(project_path, "app/src/main/java/com/iceburg/{project_name.lower()}/MainActivity.kt"), "w") as f:
            f.write(main_activity)
        
        return MobileProject(
            project_name=project_name,
            project_path=project_path,
            platform=MobilePlatform.ANDROID_STUDIO,
            features=features or [],
            metadata={"description": description}
        )
    
    async def _create_react_native_structure(self, project_path: str):
        """Create React Native project structure."""
        directories = [
            "src/screens",
            "src/components",
            "src/navigation",
            "src/utils",
            "android",
            "ios"
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(project_path, directory), exist_ok=True)
    
    async def _create_android_structure(self, project_path: str):
        """Create Android project structure."""
        directories = [
            "app/src/main/java/com/iceburg",
            "app/src/main/res/layout",
            "app/src/main/res/values",
            "app/src/main/res/drawable"
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(project_path, directory), exist_ok=True)
    
    async def _generate_react_native_screens(self, project_path: str):
        """Generate React Native screen components."""
        # HomeScreen.js
        home_screen = """import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

export default function HomeScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>ICEBURG Mobile App</Text>
      <TouchableOpacity 
        style={styles.button}
        onPress={() => navigation.navigate('Detail')}
      >
        <Text style={styles.buttonText}>Go to Detail</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
  },
});
"""
        
        with open(os.path.join(project_path, "src/screens/HomeScreen.js"), "w") as f:
            f.write(home_screen)
        
        # DetailScreen.js
        detail_screen = """import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

export default function DetailScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Detail Screen</Text>
      <Text style={styles.description}>
        This is a detail screen generated by ICEBURG.
      </Text>
      <TouchableOpacity 
        style={styles.button}
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.buttonText}>Go Back</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  description: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 30,
  },
  button: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
  },
});
"""
        
        with open(os.path.join(project_path, "src/screens/DetailScreen.js"), "w") as f:
            f.write(detail_screen)


# Convenience functions
async def create_mobile_template(config: Dict[str, Any] = None) -> MobileTemplate:
    """Create mobile template."""
    return MobileTemplate(config)


async def generate_mobile_project(project_name: str,
                                platform: MobilePlatform,
                                description: str,
                                template: MobileTemplate = None) -> MobileProject:
    """Generate mobile project."""
    if template is None:
        template = await create_mobile_template()
    
    if platform == MobilePlatform.REACT_NATIVE:
        return await template.generate_react_native_project(project_name, description)
    elif platform == MobilePlatform.ANDROID_STUDIO:
        return await template.generate_android_studio_project(project_name, description)
    else:
        raise ValueError(f"Unsupported platform: {platform}")
