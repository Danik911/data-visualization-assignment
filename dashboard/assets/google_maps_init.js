// Google Maps initialization helper
window.addEventListener('DOMContentLoaded', function() {
  // Create global variable for Map data
  window.googleMapData = null;
  console.log('Google Maps initialization script loaded');
  
  // Verify that the Google Maps API is available
  function checkGoogleMapsApi() {
    if (typeof google === 'undefined' || typeof google.maps === 'undefined') {
      console.error('Google Maps API not loaded');
      const mapContainer = document.getElementById('google-price-map-container');
      if (mapContainer) {
        mapContainer.innerHTML = `
          <div class="alert alert-danger">
            <strong>Error:</strong> Google Maps API failed to load. 
            Please check your API key and internet connection.
          </div>
        `;
      }
      return false;
    }
    console.log('Google Maps API is loaded');
    return true;
  }
  
  // Function to initialize map when data is available
  window.initGoogleMap = function(data) {
    console.log('Initializing Google Maps with data');
    
    // Wait for container to be ready
    const waitForContainer = setInterval(function() {
      const mapContainer = document.getElementById('google-price-map-container');
      if (mapContainer) {
        clearInterval(waitForContainer);
        
        // Create the map
        try {
          if (!checkGoogleMapsApi()) {
            return;
          }
          
          const mapData = typeof data === 'string' ? JSON.parse(data) : data;
          
          // Check for errors in the data
          if (mapData.error) {
            console.error('Map data error:', mapData.error);
            mapContainer.innerHTML = `<div class="alert alert-warning">Error: ${mapData.error}</div>`;
            return;
          }
          
          if (!mapData || !mapData.center || !mapData.data) {
            console.error('Invalid map data structure:', mapData);
            mapContainer.innerHTML = `<div class="alert alert-warning">Error: Invalid map data format</div>`;
            return;
          }
          
          // Initialize map if it doesn't exist yet or if it needs to be recreated
          if (!window.googleMap) {
            console.log(`Creating new Google Map centered at: ${mapData.center.lat}, ${mapData.center.lng}`);
            
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
          } else {
            // Update center if map already exists
            window.googleMap.setCenter(mapData.center);
            console.log('Using existing map object');
          }
          
          // Clear existing markers if any
          if (window.mapMarkers && window.mapMarkers.length > 0) {
            for (let marker of window.mapMarkers) {
              marker.setMap(null);
            }
          }
          
          // Store new markers
          window.mapMarkers = [];
          
          // Add markers for properties
          if (mapData.data && Array.isArray(mapData.data) && mapData.data.length > 0) {
            // Get price range from data statistics if available, otherwise calculate
            let minPrice, maxPrice;
            if (mapData.stats && typeof mapData.stats.price_min !== 'undefined' && 
                typeof mapData.stats.price_max !== 'undefined') {
              minPrice = mapData.stats.price_min;
              maxPrice = mapData.stats.price_max;
              console.log(`Using provided price range: $${minPrice} - $${maxPrice}`);
            } else {
              // Calculate price range from data points
              const prices = mapData.data.map(item => item.Sale_Price);
              minPrice = Math.min(...prices);
              maxPrice = Math.max(...prices);
              console.log(`Calculated price range: $${minPrice} - $${maxPrice}`);
            }
            
            const priceRange = maxPrice - minPrice;
            
            console.log(`Adding ${mapData.data.length} markers to map`);
            
            // Add markers for each property
            for (let property of mapData.data) {
              // Skip invalid data points
              if (typeof property.Latitude !== 'number' || typeof property.Longitude !== 'number' || 
                  typeof property.Sale_Price !== 'number') {
                console.warn('Skipping invalid property data:', property);
                continue;
              }
              
              const normalizedPrice = (property.Sale_Price - minPrice) / priceRange;
              
              // Color gradient from green (low) to red (high)
              const hue = (1 - normalizedPrice) * 120;
              const color = `hsl(${hue}, 100%, 50%)`;
              
              // Create marker for this property
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
              
              // Create rich info window content
              let contentString = `
                <div style="padding: 8px; max-width: 300px;">
                  <h5 style="margin-top: 0; color: #2c3e50;">$${property.Sale_Price.toLocaleString()}</h5>
              `;
              
              // Add additional property details if available
              if (property.Bldg_Type) {
                contentString += `<p><b>Type:</b> ${property.Bldg_Type}</p>`;
              }
              
              if (property.Year_Built) {
                contentString += `<p><b>Year Built:</b> ${property.Year_Built}</p>`;
              }
              
              if (property.Lot_Area) {
                contentString += `<p><b>Lot Area:</b> ${property.Lot_Area.toLocaleString()} sq.ft</p>`;
              }
              
              if (property.Neighborhood) {
                contentString += `<p><b>Neighborhood:</b> ${property.Neighborhood}</p>`;
              }
              
              contentString += `
                  <p><b>Location:</b> ${property.Latitude.toFixed(6)}, ${property.Longitude.toFixed(6)}</p>
                </div>
              `;
              
              // Add click event listener
              marker.addListener('click', () => {
                window.infoWindow.setContent(contentString);
                window.infoWindow.open(window.googleMap, marker);
                
                // Store click data in the hidden div
                const clickData = {
                  id: property.id || 0,
                  lat: property.Latitude,
                  lng: property.Longitude,
                  price: property.Sale_Price
                };
                
                const clickDataElement = document.getElementById('google-price-map-click-data');
                if (clickDataElement) {
                  clickDataElement.textContent = JSON.stringify(clickData);
                  
                  // Dispatch a custom event that can be used by other callbacks
                  const event = new CustomEvent('map-marker-click', { detail: clickData });
                  document.dispatchEvent(event);
                }
              });
              
              window.mapMarkers.push(marker);
            }
            
            // Create or update the legend
            if (window.mapLegend) {
              try {
                window.googleMap.controls[google.maps.ControlPosition.RIGHT_BOTTOM].removeAt(0);
              } catch (e) {
                console.warn('Error removing legend:', e);
              }
            }
            
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
            
            // Add count indicator
            const count = document.createElement('div');
            count.style.marginTop = '10px';
            count.style.fontSize = '12px';
            count.style.color = '#666';
            count.textContent = `${mapData.data.length} properties shown`;
            legend.appendChild(count);
            
            window.mapLegend = legend;
            window.googleMap.controls[google.maps.ControlPosition.RIGHT_BOTTOM].push(legend);
            
            // Update the debug div with successful status
            const debugElement = document.getElementById('google-price-map-debug');
            if (debugElement) {
              debugElement.textContent = `Map loaded with ${mapData.data.length} properties`;
              debugElement.style.display = 'block';
            }
          } else {
            console.warn('No map data points available');
            mapContainer.innerHTML += `<div class="alert alert-warning" style="position: absolute; top: 10px; left: 10px; z-index: 1000;">No property data available for the current filter criteria</div>`;
          }
          
          console.log('Map initialized successfully');
        } catch (error) {
          console.error('Error initializing Google Maps:', error);
          mapContainer.innerHTML = `<div class="alert alert-danger">Error initializing map: ${error.message}</div>`;
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
          console.log('Map data updated, reinitializing map');
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
      console.log('Found google-price-map-data element');
      
      // Check if it already has data
      if (dataElement.textContent) {
        console.log('Initial data found, initializing map');
        window.initGoogleMap(dataElement.textContent);
      } else {
        console.log('No initial data, waiting for updates');
      }
      
      // Observe future changes
      observer.observe(dataElement, { childList: true });
    }
  }, 100);
  
  // Check API status after a delay to ensure the API has time to load
  setTimeout(checkGoogleMapsApi, 2000);
});