import { Icon } from './lib/chakra';
import { MdHome, MdAutoAwesome } from 'react-icons/md';

// Auth Imports
import { IRoute } from './types/navigation';

const routes: IRoute[] = [
  {
    name: 'Dashboard',
    disabled: false,
    path: '/',
    icon: <Icon as={MdHome} width="20px" height="20px" color="inherit" />,
    collapse: false,
  },
  {
    name: 'AI Assistance',
    path: '/ai-assistance',
    icon: (
      <Icon as={MdAutoAwesome} width="20px" height="20px" color="inherit" />
    ),
    collapse: false,
  },
];

export default routes;
