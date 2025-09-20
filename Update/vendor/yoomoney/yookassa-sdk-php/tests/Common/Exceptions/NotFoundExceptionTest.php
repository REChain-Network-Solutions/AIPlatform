<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Exceptions\NotFoundException;

/**
 * @internal
 */
class NotFoundExceptionTest extends AbstractTestApiRequestException
{
    public function getTestInstance($message = '', $code = 0, $responseHeaders = [], $responseBody = '')
    {
        return new NotFoundException($responseHeaders, $responseBody);
    }

    public function expectedHttpCode()
    {
        return NotFoundException::HTTP_CODE;
    }
}
