'use client';
import { useSendChat } from '@/components/analytics/data';
/*eslint-disable*/

import Link from '@/components/link/Link';
import MessageBoxChat from '@/components/MessageBox';
import { ChatBody, OpenAIModel } from '@/types/types';
import {
  Button,
  Box,
  Flex,
  Icon,
  Input,
  Text,
  useColorModeValue,
} from '@chakra-ui/react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { MdAutoAwesome, MdBolt, MdEdit, MdPerson } from 'react-icons/md';

// export default function Chat(props: { apiKeyApp: string }) {
export default function Chat() {
  const [chatHistory, setChatHistory] = useState<Array<Chat>>([]);

      const scrollRef = useRef<HTMLDivElement>(null);
  // Input States
  const [inputOnSubmit, setInputOnSubmit] = useState<string>('');
  const [inputCode, setInputCode] = useState<string>('');
  // Response message
  const [outputCode, setOutputCode] = useState<string>('');
  // Loading state
  const [loading, setLoading] = useState<boolean>(false);

  // API Key
  // const [apiKey, setApiKey] = useState<string>(apiKeyApp);
  const borderColor = useColorModeValue('gray.200', 'whiteAlpha.200');
  const inputColor = useColorModeValue('navy.700', 'white');
  const brandColor = useColorModeValue('brand.500', 'white');
  const textColor = useColorModeValue('navy.700', 'white');
  const placeholderColor = useColorModeValue(
    { color: 'gray.500' },
    { color: 'whiteAlpha.600' },
  );
  const handleTranslate = async () => {
    if (inputCode == '') return;
    setInputOnSubmit(inputCode);
    setOutputCode(' ');
    setLoading(true);
    const body: ChatBody = {
      message: inputCode,
      session_id: 'user123',
      include_analysis: true,
    };
    // -------------- Fetch --------------

    const response = await useSendChat(body);
    if (response.error) {
      setLoading(false);
      if (response) {
        alert(
          'Something went wrong, please try again later'
        );
      }
      return;
    }

    const data = response.response;
    saveChatHistory({input: inputCode, output: data});
    setOutputCode(data);
    // scroll to bottom
    setInputCode('');
    setLoading(false);
  };

  type Chat = {
    input: string,
    output:string
  }
  const saveChatHistory = (chat: Chat) => {
    // getting saved ls data
    const chatHistory:Array<Chat> = JSON.parse(localStorage.getItem(process.env.NEXT_PUBLIC_CHAT_KEY||'') || '[]');
    chatHistory.push(chat);
    // save to local storage
    localStorage.setItem(process.env.NEXT_PUBLIC_CHAT_KEY||'', JSON.stringify(chatHistory));
  }

  const getAllChatHistory = ():void => {
    const chatHistory = JSON.parse(localStorage.getItem(process.env.NEXT_PUBLIC_CHAT_KEY||'') || '[]');
    setChatHistory(chatHistory);
  }
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

  useLayoutEffect(() => {
    getAllChatHistory();
    setTimeout(() => {

        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, 1000)
  }, [outputCode])

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
        direction="column"
        mx="auto"
        w={{ base: '100%', md: '100%', xl: '100%' }}
        minH={{ base: '75vh', '2xl': '85vh' }}
        maxW="1000px"
      >
        {/* Model Change */}
        {chatHistory.length == 0 && <Flex direction={'column'} w="00%" mb={outputCode ? '20px' : 'auto'}>
          {/* <Flex
            mx="auto"
            zIndex="2"
            w="max-content"
            mb="20px"
            borderRadius="60px"
          >
          </Flex> */}
        </Flex>}
        {/* Main Box */}
        <Box
          w="100%"
          h={'70vh'}
          overflow={'scroll'}
          ref={scrollRef}
        >
        {chatHistory.map((elem, index) => (
          <Flex
          direction={'column'}
          mx="auto"
          mb={'auto'}
          mt='2'
          display={chatHistory ? 'flex' : 'none'}
          key={`chat_id_${index}`}>
          <Flex w="100%" align={'center'} mb="10px">
            <Flex
              borderRadius="full"
              justify="center"
              align="center"
              bg={'transparent'}
              border="1px solid"
              borderColor={borderColor}
              me="20px"
              h="40px"
              minH="40px"
              minW="40px"
            >
              <Icon
                as={MdPerson}
                width="20px"
                height="20px"
                color={brandColor}
              />
            </Flex>
            <Flex
              p="8px"
              border="1px solid"
              borderColor={borderColor}
              borderRadius="14px"
              w="100%"
              zIndex={'2'}
            >
              <Text
                color={textColor}
                fontWeight="600"
                // fontSize={{ base: 'sm', md: 'md' }}
                fontSize={'xs'}
                lineHeight={{ base: '24px', md: '26px' }}
              >
                {elem.input}
              </Text>
              {/* <Icon
                cursor="pointer"
                as={MdEdit}
                ms="auto"
                width="20px"
                height="20px"
                color={gray}
              /> */}
            </Flex>
          </Flex>
          <Flex w="100%">
            <Flex
              borderRadius="full"
              justify="center"
              align="center"
              bg={'linear-gradient(15.46deg, #4A25E1 26.3%, #7B5AFF 86.4%)'}
              me="20px"
              h="40px"
              minH="40px"
              minW="40px"
            >
              <Icon
                as={MdAutoAwesome}
                width="20px"
                height="20px"
                color="white"
              />
            </Flex>
            <MessageBoxChat output={elem.output} />
          </Flex></Flex>
        ))}
        </Box>
        {/* Chat Input */}
        <Flex
          ms={{ base: '0px', xl: '60px' }}
          mt="0"
          justifySelf={'flex-end'}
        >
          <Input
            minH="54px"
            h="100%"
            border="1px solid"
            borderColor={borderColor}
            borderRadius="45px"
            p="15px 20px"
            me="10px"
            fontSize="sm"
            fontWeight="500"
            _focus={{ borderColor: 'none' }}
            color={inputColor}
            _placeholder={placeholderColor}
            placeholder="Type your message here..."
            value={inputCode}
            onChange={handleChange}
          />
          <Button
            variant="primary"
            py="20px"
            px="16px"
            fontSize="sm"
            borderRadius="45px"
            ms="auto"
            w={{ base: '160px', md: '210px' }}
            h="54px"
            _hover={{
              boxShadow:
                '0px 21px 27px -10px rgba(96, 60, 255, 0.48) !important',
              bg: 'linear-gradient(15.46deg, #4A25E1 26.3%, #7B5AFF 86.4%) !important',
              _disabled: {
                bg: 'linear-gradient(15.46deg, #4A25E1 26.3%, #7B5AFF 86.4%)',
              },
            }}
            onClick={handleTranslate}
            isLoading={loading ? true : false}
          >
            Submit
          </Button>
        </Flex>
      </Flex>
    </Flex>
  );
}
