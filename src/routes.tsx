import { Icon } from './lib/chakra';
import {
  MdHome,
  MdAutoAwesome,
} from 'react-icons/md';

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
  // --- Admin Pages ---
  // {
  //   name: 'Admin Pages',
  //   disabled: true,
  //   path: '/admin',
  //   icon: <Icon as={MdLock} width="20px" height="20px" color="inherit" />,
  //   collapse: true,
  //   items: [
  //     {
  //       name: 'All Templates',
  //       layout: '/admin',
  //       path: '/all-admin-templates',
  //     },
  //     {
  //       name: 'New Template',
  //       layout: '/admin',
  //       path: '/new-template',
  //     },
  //     {
  //       name: 'Edit Template',
  //       layout: '/admin',
  //       path: '/edit-template',
  //     },
  //     {
  //       name: 'Users Overview',
  //       layout: '/admin',
  //       path: '/overview',
  //     },
  //   ],
  // },
  // -------------- Prompt Pages --------------
  // {
  //   name: 'Essay Generator',
  //   disabled: true,
  //   path: '/essay',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'Content Simplifier',
  //   disabled: true,
  //   path: '/simplifier',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'Product Description',
  //   disabled: true,
  //   path: '/product-description',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'Email Enhancer',
  //   disabled: true,
  //   path: '/email-enhancer',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'LinkedIn Message',
  //   disabled: true,
  //   path: '/linkedin-message',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'Instagram Caption',
  //   disabled: true,
  //   path: '/caption',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'FAQs Content',
  //   disabled: true,
  //   path: '/faq',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'Product Name Generator',
  //   disabled: true,
  //   path: '/name-generator',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'SEO Keywords',
  //   disabled: true,
  //   path: '/seo-keywords',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'Review Responder',
  //   disabled: true,
  //   path: '/review-responder',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'Business Idea Generator',
  //   disabled: true,
  //   path: '/business-generator',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
  // {
  //   name: 'Article Generator',
  //   disabled: true,
  //   path: '/article',
  //   icon: <Icon as={IoMdPerson} width="20px" height="20px" color="inherit" />,
  //   invisible: true,
  //   collapse: false,
  // },
];

export default routes;
