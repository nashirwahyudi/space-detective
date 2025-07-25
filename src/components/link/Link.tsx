'use client';
import { ButtonProps } from '@chakra-ui/react';
import NextLink, { LinkProps as NextLinkProps } from 'next/link';

import { Button } from '@chakra-ui/react';

type LinkProps = ButtonProps & NextLinkProps;

function Link({ href, children, ...props }: LinkProps) {
  return (
    <NextLink href={href} passHref>
      <Button {...props}>{children}</Button>
    </NextLink>
  );
}

export default Link;
