'use client';
/*eslint-disable*/

import { ChatBody, OpenAIModel } from '@/types/types';
// import Map from '@/components/statistic/Map';
import MapProvider from '@/lib/maplibre/profider';
import {
  Box,
  Card,
  Flex,
  Table,
  TableCaption,
  TableContainer,
  Tbody,
  Td,
  Tfoot,
  Th,
  Thead,
  Tr,
  useColorModeValue,
} from '@chakra-ui/react';
import { useEffect, useRef, useState } from 'react';

// export default function Chat(props: { apiKeyApp: string }) {
export default function Dashboard() {
  // reg
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  // Input States
  const [inputOnSubmit, setInputOnSubmit] = useState<string>('');
  const [inputCode, setInputCode] = useState<string>('');
  // Response message
  const [outputCode, setOutputCode] = useState<string>('');
  // ChatGPT model
  const [model, setModel] = useState<OpenAIModel>('gpt-4o');
  // Loading state
  const [loading, setLoading] = useState<boolean>(false);

  // API Key
  // const [apiKey, setApiKey] = useState<string>(apiKeyApp);
  const borderColor = useColorModeValue('gray.200', 'whiteAlpha.200');
  const inputColor = useColorModeValue('navy.700', 'white');
  const iconColor = useColorModeValue('brand.500', 'white');
  const bgIcon = useColorModeValue(
    'linear-gradient(180deg, #FBFBFF 0%, #CACAFF 100%)',
    'whiteAlpha.200',
  );
  const brandColor = useColorModeValue('brand.500', 'white');
  const buttonBg = useColorModeValue('white', 'whiteAlpha.100');
  const gray = useColorModeValue('gray.500', 'white');
  const buttonShadow = useColorModeValue(
    '14px 27px 45px rgba(112, 144, 176, 0.2)',
    'none',
  );
  const textColor = useColorModeValue('navy.700', 'white');
  const placeholderColor = useColorModeValue(
    { color: 'gray.500' },
    { color: 'whiteAlpha.600' },
  );
  const handleTranslate = async () => {
    let apiKey = localStorage.getItem('apiKey');
    setInputOnSubmit(inputCode);

    // Chat post conditions(maximum number of characters, valid message etc.)
    const maxCodeLength = model === 'gpt-4o' ? 700 : 700;

    if (!apiKey?.includes('sk-')) {
      alert('Please enter an API key.');
      return;
    }

    if (!inputCode) {
      alert('Please enter your message.');
      return;
    }

    if (inputCode.length > maxCodeLength) {
      alert(
        `Please enter code less than ${maxCodeLength} characters. You are currently at ${inputCode.length} characters.`,
      );
      return;
    }
    setOutputCode(' ');
    setLoading(true);
    const controller = new AbortController();
    const body: ChatBody = {
      inputCode,
      model,
      apiKey,
    };

    // -------------- Fetch --------------
    const response = await fetch('./api/chatAPI', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      setLoading(false);
      if (response) {
        alert(
          'Something went wrong went fetching from the API. Make sure to use a valid API key.',
        );
      }
      return;
    }

    const data = response.body;

    if (!data) {
      setLoading(false);
      alert('Something went wrong');
      return;
    }

    const reader = data.getReader();
    const decoder = new TextDecoder();
    let done = false;

    while (!done) {
      setLoading(true);
      const { value, done: doneReading } = await reader.read();
      done = doneReading;
      const chunkValue = decoder.decode(value);
      setOutputCode((prevCode) => prevCode + chunkValue);
    }

    setLoading(false);
  };
  // -------------- Copy Response --------------
  // const copyToClipboard = (text: string) => {
  //   const el = document.createElement('textarea');
  //   el.value = text;
  //   document.body.appendChild(el);
  //   el.select();
  //   document.execCommand('copy');
  //   document.body.removeChild(el);
  // };

  // *** Initializing apiKey with .env.local value
  // useEffect(() => {
  // ENV file verison
  // const apiKeyENV = process.env.NEXT_PUBLIC_OPENAI_API_KEY
  // if (apiKey === undefined || null) {
  //   setApiKey(apiKeyENV)
  // }
  // }, [])

  const handleChange = (Event: any) => {
    setInputCode(Event.target.value);
  };

  return (
    <Flex
      w="100%"
      pt={{ base: '70px', md: '0px' }}
      direction="column"
      position="relative"
    >
      <Flex
        direction="row"
        mx="auto"
        w={{ base: '100%', md: '100%', xl: '100%' }}
        minH={{ base: '75vh', '2xl': '85vh' }}
        maxW="1000px"
      >
        <Flex direction={'row'} w="50%" h={{ base: '75vh', '2xl': '85vh' }} mb={'auto'} marginEnd={5}>
          <Card
            display={'flex'}
            px="22px !important"
            py="22px !important"
            w={'100%'}
            h={'100%'}
            color={textColor}
            fontSize={{ base: 'sm', md: 'md' }}
            lineHeight={{ base: '24px', md: '26px' }}
            fontWeight="500"
          >
            {/* <Map/> */}
            <Box 
              id="map-container"
              ref={mapContainerRef}
              w={'100%'}
              h={'100%'}
              display='absolute'
              inset={0}>
              <MapProvider 
              mapContainerRef={mapContainerRef}
              initialViewState={{
                longitude: -122.4194,
                latitude: 37.7749,
                zoom: 10,
              }}></MapProvider>
            </Box>
          </Card>
        </Flex>
        <Flex direction={'column'} h={{ base: '75vh', '2xl': '85vh' }} w="50%" mb={'auto'} gap='5'>
          <Card
            display={'flex'}
            px="22px !important"
            w={'100%'}
            py="22px !important"
            h={{base: '50%'}}
            color={textColor}
            fontSize={{ base: 'sm', md: 'md' }}
            lineHeight={{ base: '24px', md: '26px' }}
            fontWeight="500"
          >
            {/* Tabel checklist */}
            <TableContainer>
              <Table variant='striped' size='md'>
                <Thead>
                  <Tr>
                    <Th>Highlight</Th>
                    <Th isNumeric>2022</Th>
                    <Th isNumeric>2023</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  <Tr>
                    <Td>Provinsi A</Td>
                    <Td isNumeric>0.7</Td>
                    <Td isNumeric>0.76</Td>
                  </Tr>
                  <Tr>
                    <Td>Provinsi B</Td>
                    <Td isNumeric>0.97</Td>
                    <Td isNumeric>0.62</Td>
                  </Tr>
                  <Tr>
                    <Td>Provinsi C</Td>
                    <Td isNumeric>0.34</Td>
                    <Td isNumeric>0.63</Td>
                  </Tr>
                </Tbody>
              </Table>
            </TableContainer>
          </Card>
          <Card
            display={'flex'}
            px="22px !important"
            py="22px !important"
            h={{base: '50%'}}
            w={'100%'}
            color={textColor}
            fontSize={{ base: 'sm', md: 'md' }}
            lineHeight={{ base: '24px', md: '26px' }}
            fontWeight="500"
          >
            Shap value
          </Card>
        </Flex>
      </Flex>
    </Flex>
  );
}
