const primitivesDiv = document.getElementById('primitives-list');
const ROOT = 'http://127.0.0.1:5000';

async function fetchPrimitives() {
  const result = await fetch(`${ROOT}/primitives`, {
    method: 'GET'
  });
  const jsonData = await result.json();
  console.log(jsonData);

  primitivesDiv.innerHTML = jsonData.primitives;
}

async function getTask() {
  const result = await fetch(`${ROOT}/dataset/0`, {
    method: 'GET'
  });
  const jsonData = await result.json();
  console.log(jsonData);
}