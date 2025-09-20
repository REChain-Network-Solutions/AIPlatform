<?php

/*
* The MIT License
*
* Copyright (c) 2025 "YooMoney", NBСO LLC
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

namespace Tests\YooKassa\Request\Payments\PaymentData;

use InvalidArgumentException;
use YooKassa\Helpers\Random;
use YooKassa\Request\Payments\PaymentData\PaymentDataGooglePay;

/**
 * AbstractTestPaymentDataGooglePay
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
abstract class AbstractTestPaymentDataGooglePay extends AbstractTestPaymentData
{
    /**
     * @dataProvider validPaymentDataDataProvider
     */
    public function testGetSetPaymentMethodToken(string $data): void
    {
        /** @var PaymentDataGooglePay $instance */
        $instance = $this->getTestInstance();

        $instance->setPaymentMethodToken($data);
        self::assertEquals($data, $instance->getPaymentMethodToken());

        $instance = $this->getTestInstance();
        $instance->paymentMethodToken = $data;
        self::assertEquals($data, $instance->getPaymentMethodToken());
    }

    /**
     * @dataProvider validPaymentDataDataProvider
     */
    public function testGetSetGoogleTransactionId(string $data): void
    {
        /** @var PaymentDataGooglePay $instance */
        $instance = $this->getTestInstance();
        $instance->setGoogleTransactionId($data);
        self::assertEquals($data, $instance->getGoogleTransactionId());

        $instance = $this->getTestInstance();
        $instance->googleTransactionId = $data;
        self::assertEquals($data, $instance->getGoogleTransactionId());
    }

    /**
     * @dataProvider invalidPaymentDataDataProvider
     *
     * @param mixed $data
     */
    public function testSetPaymentMethodToken(mixed $data): void
    {
        $this->expectException(InvalidArgumentException::class);

        /** @var PaymentDataGooglePay $instance */
        $instance = $this->getTestInstance();
        $instance->setPaymentMethodToken($data);
    }

    /**
     * @dataProvider invalidPaymentDataDataProvider
     *
     * @param mixed $data
     */
    public function testSetGoogleTransactionId(mixed $data): void
    {
        $this->expectException(InvalidArgumentException::class);

        /** @var PaymentDataGooglePay $instance */
        $instance = $this->getTestInstance();
        $instance->setGoogleTransactionId($data);
    }

    public function validPaymentDataDataProvider(): array
    {
        return [
            [Random::str(256)],
            [Random::str(1024)],
        ];
    }

    public function invalidPaymentDataDataProvider(): array
    {
        return [
            [null],
            [''],
            [false],
        ];
    }
}
