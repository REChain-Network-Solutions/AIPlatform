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

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\PaymentData\PaymentDataGooglePay;

/**
 * PaymentDataGooglePayTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentDataGooglePayTest extends AbstractTestCase
{
    protected PaymentDataGooglePay $object;

    /**
     * @return PaymentDataGooglePay
     */
    protected function getTestInstance(): PaymentDataGooglePay
    {
        return new PaymentDataGooglePay();
    }

    /**
     * @return void
     */
    public function testPaymentDataGooglePayClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentDataGooglePay::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentDataGooglePay::class));
        $this->assertInstanceOf(PaymentDataGooglePay::class, $this->object);
    }

    /**
     * Test property "payment_method_token"
     * @dataProvider validPaymentMethodTokenDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentMethodToken(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentMethodToken());
        self::assertEmpty($instance->payment_method_token);
        $instance->setPaymentMethodToken($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentMethodToken()->toArray() : $instance->getPaymentMethodToken());
        self::assertEquals($value, is_array($value) ? $instance->payment_method_token->toArray() : $instance->payment_method_token);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentMethodToken());
            self::assertNotNull($instance->payment_method_token);
        }
    }

    /**
     * Test invalid property "payment_method_token"
     * @dataProvider invalidPaymentMethodTokenDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentMethodToken(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentMethodToken($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentMethodTokenDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_method_token'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentMethodTokenDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_method_token'));
    }

    /**
     * Test property "google_transaction_id"
     * @dataProvider validGoogleTransactionIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testGoogleTransactionId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getGoogleTransactionId());
        self::assertEmpty($instance->google_transaction_id);
        $instance->setGoogleTransactionId($value);
        self::assertEquals($value, is_array($value) ? $instance->getGoogleTransactionId()->toArray() : $instance->getGoogleTransactionId());
        self::assertEquals($value, is_array($value) ? $instance->google_transaction_id->toArray() : $instance->google_transaction_id);
        if (!empty($value)) {
            self::assertNotNull($instance->getGoogleTransactionId());
            self::assertNotNull($instance->google_transaction_id);
        }
    }

    /**
     * Test invalid property "google_transaction_id"
     * @dataProvider invalidGoogleTransactionIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidGoogleTransactionId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setGoogleTransactionId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validGoogleTransactionIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_google_transaction_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidGoogleTransactionIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_google_transaction_id'));
    }
}
