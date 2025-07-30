// map-context.ts
import { createContext, useContext } from 'react';

interface MapContextType {
  map: maplibregl.Map;
  loaded: boolean;
}

export const MapContext = createContext<MapContextType | null>(null);

export const useMap = () => {
  const context = useContext(MapContext);
  if (!context) {
    throw new Error('useMap must be used within a MapProvider');
  }
  return context;
};
