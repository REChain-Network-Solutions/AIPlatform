<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Exceptions\InternalServerError;

/**
 * @internal
 */
class InternalServerErrorTest extends AbstractTestApiRequestException
{
    public function getTestInstance($message = '', $code = 0, $responseHeaders = [], $responseBody = '')
    {
        return new InternalServerError($responseHeaders, $responseBody);
    }

    public function expectedHttpCode()
    {
        return InternalServerError::HTTP_CODE;
    }
}
