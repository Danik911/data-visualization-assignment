// Google Maps initialization helper
window.addEventListener('DOMContentLoaded', function() {
  // Create global variable for Map data
  window.googleMapData = null;
  
  // Function to initialize map when data is available
  window.initGoogleMap = function(data) {
    console.log('Initializing Google Maps with data:', data);
    
    // Wait for container to be ready
    const waitForContainer = setInterval(function() {
      const mapContainer = document.getElementById('google-price-map-container');
      if (mapContainer) {
        clearInterval(waitForContainer);
        
        // Create the map
        try {
          const mapData = typeof data === 'string' ? JSON.parse(data) : data;
          
          if (!mapData || !mapData.center) {
            console.error('Invalid map data structure:', mapData);
            return;
          }
          
          // Initialize map
          window.googleMap = new google.maps.Map(mapContainer, {
            center: mapData.center,
            zoom: mapData.zoom || 12,
            mapTypeId: google.maps.MapTypeId.ROADMAP,
            mapTypeControl: true,
            streetViewControl: true,
            fullscreenControl: true
          });
          
          // Create info window for markers
          window.infoWindow = new google.maps.InfoWindow();
          
          // Add markers
          if (mapData.data && Array.isArray(mapData.data) && mapData.data.length > 0) {
            // Store markers
            window.mapMarkers = [];
            
            // Price range for color calculation
            const prices = mapData.data.map(item => item.Sale_Price);
            const minPrice = Math.min(...prices);
            const maxPrice = Math.max(...prices);
            const priceRange = maxPrice - minPrice;
            
            console.log(`Adding ${mapData.data.length} markers to map`);
            
            // Add markers for each property
            for (let property of mapData.data) {
              const normalizedPrice = (property.Sale_Price - minPrice) / priceRange;
              
              // Color gradient from green (low) to red (high)
              const hue = (1 - normalizedPrice) * 120;
              const color = `hsl(${hue}, 100%, 50%)`;
              
              const marker = new google.maps.Marker({
                position: {
                  lat: property.Latitude,
                  lng: property.Longitude
                },
                map: window.googleMap,
                title: `$${property.Sale_Price.toLocaleString()}`,
                icon: {
                  path: google.maps.SymbolPath.CIRCLE,
                  fillColor: color,
                  fillOpacity: 0.7,
                  strokeColor: 'white',
                  strokeWeight: 1,
                  scale: 8 + (normalizedPrice * 8) // Size 8-16 based on price
                }
              });
              
              // Create info window content
              const contentString = `
                <div style="padding: 8px;">
                  <h5 style="margin-top: 0;">$${property.Sale_Price.toLocaleString()}</h5>
                  ${property.Bldg_Type ? `<p><b>Type:</b> ${property.Bldg_Type}</p>` : ''}
                  ${property.Year_Built ? `<p><b>Year Built:</b> ${property.Year_Built}</p>` : ''}
                  <p><b>Location:</b> ${property.Latitude.toFixed(6)}, ${property.Longitude.toFixed(6)}</p>
                </div>
              `;
              
              // Add click event listener
              marker.addListener('click', () => {
                window.infoWindow.setContent(contentString);
                window.infoWindow.open(window.googleMap, marker);
              });
              
              window.mapMarkers.push(marker);
            }
            
            // Create legend
            const legend = document.createElement('div');
            legend.style.backgroundColor = 'white';
            legend.style.padding = '10px';
            legend.style.margin = '10px';
            legend.style.borderRadius = '5px';
            legend.style.boxShadow = '0 2px 6px rgba(0,0,0,.3)';
            
            const title = document.createElement('h4');
            title.textContent = 'Price Legend';
            title.style.margin = '0 0 8px';
            legend.appendChild(title);
            
            // Add legend items
            const steps = 5;
            for (let i = 0; i < steps; i++) {
              const item = document.createElement('div');
              item.style.display = 'flex';
              item.style.alignItems = 'center';
              item.style.marginBottom = '5px';
              
              const normalizedValue = i / (steps - 1);
              const hue = (1 - normalizedValue) * 120;
              const color = `hsl(${hue}, 100%, 50%)`;
              
              const colorBox = document.createElement('div');
              colorBox.style.width = '20px';
              colorBox.style.height = '20px';
              colorBox.style.backgroundColor = color;
              colorBox.style.marginRight = '8px';
              colorBox.style.border = '1px solid #ccc';
              item.appendChild(colorBox);
              
              const price = minPrice + (priceRange * normalizedValue);
              const label = document.createElement('span');
              label.textContent = `$${Math.round(price).toLocaleString()}`;
              item.appendChild(label);
              
              legend.appendChild(item);
            }
            
            window.googleMap.controls[google.maps.ControlPosition.RIGHT_BOTTOM].push(legend);
          }
          
          console.log('Map initialized successfully');
        } catch (error) {
          console.error('Error initializing Google Maps:', error);
        }
      }
    }, 100);
  };
  
  // Observer to watch for data updates
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.type === 'childList' && mutation.target.id === 'google-price-map-data') {
        const data = mutation.target.textContent;
        if (data && data !== window.googleMapData) {
          window.googleMapData = data;
          window.initGoogleMap(data);
        }
      }
    });
  });
  
  // Wait for the map data element to exist, then start observing it
  const waitForDataElement = setInterval(function() {
    const dataElement = document.getElementById('google-price-map-data');
    if (dataElement) {
      clearInterval(waitForDataElement);
      
      // Check if it already has data
      if (dataElement.textContent) {
        window.initGoogleMap(dataElement.textContent);
      }
      
      // Observe future changes
      observer.observe(dataElement, { childList: true });
    }
  }, 100);
});