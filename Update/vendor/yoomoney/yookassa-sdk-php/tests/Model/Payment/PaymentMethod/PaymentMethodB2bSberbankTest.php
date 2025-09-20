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

namespace Tests\YooKassa\Model\Payment\PaymentMethod;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\PaymentMethod\PaymentMethodB2bSberbank;

/**
 * PaymentMethodB2bSberbankTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentMethodB2bSberbankTest extends AbstractTestCase
{
    protected PaymentMethodB2bSberbank $object;

    /**
     * @return PaymentMethodB2bSberbank
     */
    protected function getTestInstance(): PaymentMethodB2bSberbank
    {
        return new PaymentMethodB2bSberbank();
    }

    /**
     * @return void
     */
    public function testPaymentMethodB2bSberbankClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentMethodB2bSberbank::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentMethodB2bSberbank::class));
        $this->assertInstanceOf(PaymentMethodB2bSberbank::class, $this->object);
    }

    /**
     * Test property "type"
     *
     * @return void
     * @throws Exception
     */
    public function testType(): void
    {
        $instance = $this->getTestInstance();
        self::assertContains($instance->getType(), ['b2b_sberbank']);
        self::assertContains($instance->type, ['b2b_sberbank']);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
    }

    /**
     * Test invalid property "type"
     * @dataProvider invalidTypeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidType(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setType($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }

    /**
     * Test property "payment_purpose"
     * @dataProvider validPaymentPurposeDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentPurpose(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPaymentPurpose($value);
        self::assertNotNull($instance->getPaymentPurpose());
        self::assertNotNull($instance->payment_purpose);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentPurpose()->toArray() : $instance->getPaymentPurpose());
        self::assertEquals($value, is_array($value) ? $instance->payment_purpose->toArray() : $instance->payment_purpose);
        self::assertMatchesRegularExpression("/.{1,210}/", $instance->getPaymentPurpose());
        self::assertMatchesRegularExpression("/.{1,210}/", $instance->payment_purpose);
    }

    /**
     * Test invalid property "payment_purpose"
     * @dataProvider invalidPaymentPurposeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentPurpose(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentPurpose($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentPurposeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_purpose'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentPurposeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_purpose'));
    }

    /**
     * Test property "vat_data"
     * @dataProvider validVatDataDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testVatData(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setVatData($value);
        self::assertNotNull($instance->getVatData());
        self::assertNotNull($instance->vat_data);
        self::assertEquals($value, is_array($value) ? $instance->getVatData()->toArray() : $instance->getVatData());
        self::assertEquals($value, is_array($value) ? $instance->vat_data->toArray() : $instance->vat_data);
    }

    /**
     * Test invalid property "vat_data"
     * @dataProvider invalidVatDataDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidVatData(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setVatData($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validVatDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_vat_data'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidVatDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_vat_data'));
    }

    /**
     * Test property "payer_bank_details"
     * @dataProvider validPayerBankDetailsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPayerBankDetails(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPayerBankDetails());
        self::assertEmpty($instance->payer_bank_details);
        $instance->setPayerBankDetails($value);
        self::assertEquals($value, is_array($value) ? $instance->getPayerBankDetails()->toArray() : $instance->getPayerBankDetails());
        self::assertEquals($value, is_array($value) ? $instance->payer_bank_details->toArray() : $instance->payer_bank_details);
        if (!empty($value)) {
            self::assertNotNull($instance->getPayerBankDetails());
            self::assertNotNull($instance->payer_bank_details);
        }
    }

    /**
     * Test invalid property "payer_bank_details"
     * @dataProvider invalidPayerBankDetailsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPayerBankDetails(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPayerBankDetails($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPayerBankDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payer_bank_details'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPayerBankDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payer_bank_details'));
    }
}
