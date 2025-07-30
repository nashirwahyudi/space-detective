'use client';
import { useEffect } from 'react';
import { useMap } from '@/contexts/MapContext';
import { useFetchGeoJsonLayer } from '@/components/analytics/data';

export default function H3Layer(props: {
  h3_index: any;
  iddesa: any;
  idkec: any;
  idkab: any;
}) {
  const { map, loaded } = useMap();

  const loadGeoJSON = async () => {
      let params = new URLSearchParams({
        h3_index: props.h3_index,
        iddesa: props.iddesa,
        idkec: props.idkec,
        idkab: props.idkab,
      });
      const res = await useFetchGeoJsonLayer(params);

      const sourceId = 'pg-geojson';
      const layerId = 'pg-layer';
      const borderlayerId = 'pg-layer-1';

        // Remove old layer/source
        if (map && map.getLayer(layerId)) map.removeLayer(layerId);
        if (map && map.getLayer(borderlayerId)) map.removeLayer(borderlayerId);
        if (map && map.getSource(sourceId)) map.removeSource(sourceId);
        // if (map && map.getSource(bordersourceId)) map.removeSource(bordersourceId);

        map.addSource(sourceId, {
        type: 'geojson',
        data: res.data,
      });
      map.addLayer({
        id: layerId,
        type: 'fill',
        source: sourceId,
        paint: {
          'fill-opacity': 0.6,
          'fill-color': [
            'interpolate',
            ['linear'],
            ['get', 'anomaly_score_probability'],
            0.0,
            '#16A34A', // green
            0.2,
            '#84CC16', // lime green
            0.4,
            '#FACC15', // yellow
            0.6,
            '#FB923C', // orange
            0.8,
            '#DC2626', // red
          ],
        },
      });

      map.addLayer({
        id: borderlayerId,
        type: 'line',
        source: sourceId,
        paint: {
          'line-color': [
            'interpolate',
            ['linear'],
            ['get', 'anomaly_score_probability'],
            0.0,
            '#16A34A', // green
            0.2,
            '#84CC16', // lime green
            0.4,
            '#FACC15', // yellow
            0.6,
            '#FB923C', // orange
            0.8,
            '#DC2626', // red
          ],
          'line-width': 0.5,
        },
      });
  }

  useEffect(() => {
    if (!map || !loaded) return;

  if (map.isStyleLoaded()) {
    loadGeoJSON(); // 🔄 already loaded
  } else {
    map.once('style.load', loadGeoJSON); // ✅ wait
  }
  }, [map, props, loaded]);

  return null;
}
