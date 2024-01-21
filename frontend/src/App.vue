<template>
  <div id="app">
    <div class="upload-container">
      <h1>Image Prediction</h1>
      <input type="file" @change="handleFiles" multiple>
      <button @click="uploadFiles">Predict</button>
      <div v-if="prediction">
        <h2>Prediction Result:</h2>
        <pre>{{ prediction }}</pre>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      files: [],
      prediction: null
    };
  },
  methods: {
    handleFiles(event) {
      this.files = event.target.files;
    },
    uploadFiles() {
      const formData = new FormData();
      for (let file of this.files) {
        formData.append('files', file);
      }

      fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        body: formData,
      })
      .then(response => response.json())
      .then(data => {
        this.prediction = data;
      })
      .catch(() => {
        this.prediction = 'Error in prediction';
      });
    }
  }
};
</script>

<style>
#app {
  text-align: center;
  margin-top: 50px;
}

.upload-container {
  margin: auto;
  width: 50%;
  border: 1px solid #000;
  padding: 20px;
  box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
  transition: 0.3s;
}

.upload-container:hover {
  box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}

h1 {
  color: #333;
}

button {
  margin-top: 10px;
  padding: 10px 20px;
  background-color: #4CAF50;
  color: white;
  border: none;
  cursor: pointer;
}

button:hover {
  background-color: #45a049;
}

pre {
  text-align: left;
  color: #555;
}
</style>
