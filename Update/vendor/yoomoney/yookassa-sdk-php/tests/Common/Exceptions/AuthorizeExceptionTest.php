<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Exceptions\AuthorizeException;

/**
 * @internal
 */
class AuthorizeExceptionTest extends ApiExceptionTest
{
    public function getTestInstance($message = '', $code = 0, $responseHeaders = [], $responseBody = '')
    {
        return new AuthorizeException($message, $code, $responseHeaders, $responseBody);
    }
}
