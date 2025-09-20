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
use YooKassa\Request\Payments\PaymentData\PaymentDataApplePay;

/**
 * PaymentDataApplePayTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentDataApplePayTest extends AbstractTestCase
{
    protected PaymentDataApplePay $object;

    /**
     * @return PaymentDataApplePay
     */
    protected function getTestInstance(): PaymentDataApplePay
    {
        return new PaymentDataApplePay();
    }

    /**
     * @return void
     */
    public function testPaymentDataApplePayClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentDataApplePay::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentDataApplePay::class));
        $this->assertInstanceOf(PaymentDataApplePay::class, $this->object);
    }

    /**
     * Test property "payment_data"
     * @dataProvider validPaymentDataDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentData(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentData());
        self::assertEmpty($instance->payment_data);
        $instance->setPaymentData($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentData()->toArray() : $instance->getPaymentData());
        self::assertEquals($value, is_array($value) ? $instance->payment_data->toArray() : $instance->payment_data);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentData());
            self::assertNotNull($instance->payment_data);
        }
    }

    /**
     * Test invalid property "payment_data"
     * @dataProvider invalidPaymentDataDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentData(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentData($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_data'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_data'));
    }
}
