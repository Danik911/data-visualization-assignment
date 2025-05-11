// Ensure dash_clientside object exists
window.dash_clientside = window.dash_clientside || {};

// Assign to the dash_clientside namespace
window.dash_clientside.googleMaps = {
    initMap: function(mapDataJSON, mapContainerId) {
        console.log('RENDER_DEBUG: dash_clientside.googleMaps.initMap CALLED. Container ID:', mapContainerId);

        if (!mapContainerId) {
            console.error('RENDER_DEBUG: mapContainerId is missing!');
            return '';
        }

        const mapElement = document.getElementById(mapContainerId);
        if (!mapElement) {
            console.error('RENDER_DEBUG: Map element not found for ID:', mapContainerId);
            return '';
        }

        // Ensure the Google Maps API is loaded
        if (typeof google === 'undefined' || typeof google.maps === 'undefined') {
            console.error('RENDER_DEBUG: Google Maps API not loaded.');
            // Attempt to load it if a callback `gm_authFailure` or similar is not already defined globally by the API script.
            // This is a fallback and might not always work depending on how the main API script is loaded.
            // It's generally better to ensure the API script in index_string is correctly loaded.
            return 'Google Maps API not ready';
        }
        
        let mapData;
        if (typeof mapDataJSON === 'string') {
            try {
                mapData = JSON.parse(mapDataJSON);
            } catch (e) {
                console.error('RENDER_DEBUG: Failed to parse mapDataJSON:', e, mapDataJSON);
                mapData = { data: [], center: { lat: 42.032974, lng: -93.61969 }, zoom: 10 }; // Default on error
            }
        } else {
            mapData = mapDataJSON; // Assume it's already an object
        }

        if (!mapData || !mapData.data) {
            console.warn('RENDER_DEBUG: mapData or mapData.data is null or undefined. Using default map settings.', mapData);
            // Provide default values if mapData or mapData.data is not available
            mapData = { data: [], center: { lat: 42.032974, lng: -93.61969 }, zoom: 10 };
        }
        
        console.log('RENDER_DEBUG: Parsed mapData:', mapData);

        const defaultCenter = { lat: 42.032974, lng: -93.61969 }; // Ames, Iowa
        const defaultZoom = 12;

        let currentCenter = mapData.center && mapData.center.lat && mapData.center.lng ? mapData.center : defaultCenter;
        let currentZoom = mapData.zoom !== undefined ? mapData.zoom : defaultZoom;
        
        // Initialize the map
        // Check if a map instance already exists for this container to avoid re-initializing
        // This basic check might need to be more robust depending on how Dash re-renders
        if (!mapElement.hasOwnProperty('_googleMapInstance')) {
            console.log('RENDER_DEBUG: Initializing new Google Map instance.');
            mapElement._googleMapInstance = new google.maps.Map(mapElement, {
                center: currentCenter,
                zoom: currentZoom,
                mapTypeId: 'roadmap',
                streetViewControl: false,
                mapTypeControl: true,
                mapTypeControlOptions: {
                    style: google.maps.MapTypeControlStyle.DROPDOWN_MENU,
                    position: google.maps.ControlPosition.TOP_RIGHT,
                },
                zoomControl: true,
                zoomControlOptions: {
                    position: google.maps.ControlPosition.LEFT_BOTTOM,
                },
                fullscreenControl: false,
            });
        } else {
            console.log('RENDER_DEBUG: Using existing Google Map instance. Setting center and zoom.');
            // If map already exists, just update its center and zoom if necessary
             mapElement._googleMapInstance.setCenter(currentCenter);
             mapElement._googleMapInstance.setZoom(currentZoom);
        }
        
        const map = mapElement._googleMapInstance;

        // Clear existing markers (if any) - simple approach: store markers in an array on the mapElement
        if (mapElement._markers && mapElement._markers.length > 0) {
            console.log(`RENDER_DEBUG: Clearing ${mapElement._markers.length} existing markers.`);
            mapElement._markers.forEach(marker => marker.setMap(null));
            mapElement._markers = [];
        } else {
            mapElement._markers = [];
        }
        
        let infoWindow = mapElement._infoWindow || new google.maps.InfoWindow();
        mapElement._infoWindow = infoWindow;


        if (mapData.data && mapData.data.length > 0) {
            console.log(`RENDER_DEBUG: Adding ${mapData.data.length} new markers.`);
            mapData.data.forEach(property => {
                if (property.Latitude != null && property.Longitude != null) {
                    const marker = new google.maps.Marker({
                        position: { lat: property.Latitude, lng: property.Longitude },
                        map: map,
                        title: `Price: $${property.Sale_Price}\nType: ${property.Bldg_Type_Display}`,
                        // Potentially add custom icon based on cluster or price
                    });

                    marker.addListener('click', () => {
                        const content = `
                            <div>
                                <h6>Property Details</h6>
                                <p><strong>Price:</strong> $${property.Sale_Price.toLocaleString()}</p>
                                <p><strong>Type:</strong> ${property.Bldg_Type_Display}</p>
                                <p><strong>Area:</strong> ${property.Lot_Area} sqft</p>
                                <p><strong>Built:</strong> ${property.Year_Built}</p>
                            </div>
                        `;
                        infoWindow.setContent(content);
                        infoWindow.open(map, marker);
                    });
                    mapElement._markers.push(marker);
                } else {
                     console.warn('RENDER_DEBUG: Marker skipped due to null Latitude/Longitude:', property);
                }
            });
        } else {
            console.log('RENDER_DEBUG: No marker data provided or data is empty.');
        }

        // Return a value for the Dash Output (can be empty if the output is just a placeholder)
        return 'Map updated with ' + (mapData.data ? mapData.data.length : 0) + ' markers.';
    }
};

console.log('Google Maps init.js PARSED and dash_clientside.googleMaps defined.');

// All other previous JS code from this file is temporarily removed for this test.