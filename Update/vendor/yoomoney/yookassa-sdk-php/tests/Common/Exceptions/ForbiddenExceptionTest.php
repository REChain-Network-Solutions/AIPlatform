<?php

namespace Tests\YooKassa\Common\Exceptions;

use YooKassa\Common\Exceptions\ForbiddenException;

/**
 * @internal
 */
class ForbiddenExceptionTest extends AbstractTestApiRequestException
{
    public function getTestInstance($message = '', $code = 0, $responseHeaders = [], $responseBody = ''): ForbiddenException
    {
        return new ForbiddenException($responseHeaders, $responseBody);
    }

    public function expectedHttpCode(): int
    {
        return ForbiddenException::HTTP_CODE;
    }
}
