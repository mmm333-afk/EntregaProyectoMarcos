document.getElementById('analizar').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const resDiv = document.getElementById('resultado');
  resDiv.innerText = "Analizando...";

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      // Buscamos el texto del tweet en el código de la página de X
      const tweet = document.querySelector('article div[data-testid="tweetText"]');
      return tweet ? tweet.innerText : null;
    }
  }, async (results) => {
    const textoTweet = results[0].result;
    
    if (!textoTweet) {
      resDiv.innerText = "No veo ningún tweet abierto.";
      return;
    }

    // Enviamos el texto a tu Python (localhost:5000)
    const datos = new FormData();
    datos.append('news', textoTweet);

    try {
      const respuesta = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: datos
      });
      const html = await respuesta.text();
      
      // Limpiamos el texto para mostrar solo el resultado final
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = html;
      resDiv.innerText = tempDiv.innerText.split("---")[0].trim(); 
    } catch (e) {
      resDiv.innerText = "Error: ¿Encendiste Python?";
    }
  });
});