<?php

namespace Tests\YooKassa\Common\Exceptions;

use PHPUnit\Framework\TestCase;
use YooKassa\Common\AbstractRequest;
use YooKassa\Common\Exceptions\InvalidRequestException;
use YooKassa\Request\Payments\CreatePaymentRequest;
use YooKassa\Request\Payments\PaymentsRequest;

/**
 * @internal
 */
class InvalidRequestExceptionTest extends TestCase
{
    /**
     * @dataProvider requestObjectDataProvider
     *
     * @param mixed $requestObject
     */
    public function testGetRequestObject($requestObject): void
    {
        $instance = new InvalidRequestException($requestObject);
        if ($requestObject instanceof AbstractRequest) {
            self::assertSame($requestObject, $instance->getRequestObject());
            $message = 'Failed to build request "' . $requestObject::class . '": ""';
            self::assertEquals($message, $instance->getMessage());
        } else {
            self::assertNull($instance->getRequestObject());
            self::assertEquals($requestObject, $instance->getMessage());
        }
    }

    public static function requestObjectDataProvider()
    {
        return [
            [new PaymentsRequest()],
            [new CreatePaymentRequest()],
        ];
    }
}
