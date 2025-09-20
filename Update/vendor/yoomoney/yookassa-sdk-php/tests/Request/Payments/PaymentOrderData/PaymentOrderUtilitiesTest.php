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

namespace Tests\YooKassa\Request\Payments\PaymentOrderData;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\PaymentOrderData\PaymentOrderUtilities;

/**
 * PaymentOrderUtilitiesTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class PaymentOrderUtilitiesTest extends AbstractTestCase
{
    protected PaymentOrderUtilities $object;

    /**
    * @return PaymentOrderUtilities
    */
    protected function getTestInstance(): PaymentOrderUtilities
    {
        return new PaymentOrderUtilities();
    }

    /**
    * @return void
    */
    public function testPaymentOrderUtilitiesClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentOrderUtilities::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentOrderUtilities::class));
        $this->assertInstanceOf(PaymentOrderUtilities::class, $this->object);
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
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
        self::assertContains($instance->getType(), ['utilities']);
        self::assertContains($instance->type, ['utilities']);
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
    * Test property "amount"
    * @dataProvider validAmountDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testAmount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setAmount($value);
        self::assertNotNull($instance->getAmount());
        self::assertNotNull($instance->amount);
        self::assertEquals($value, is_array($value) ? $instance->getAmount()->toArray() : $instance->getAmount());
        self::assertEquals($value, is_array($value) ? $instance->amount->toArray() : $instance->amount);
    }

    /**
    * Test invalid property "amount"
    * @dataProvider invalidAmountDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidAmount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAmount($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_amount'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_amount'));
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
        self::assertEquals($value, $instance->getPaymentPurpose());
        self::assertEquals($value, $instance->payment_purpose);
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
    * Test property "recipient"
    * @dataProvider validRecipientDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testRecipient(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setRecipient($value);
        self::assertNotNull($instance->getRecipient());
        self::assertNotNull($instance->recipient);
        self::assertEquals($value, is_array($value) ? $instance->getRecipient()->toArray() : $instance->getRecipient());
        self::assertEquals($value, is_array($value) ? $instance->recipient->toArray() : $instance->recipient);
    }

    /**
    * Test invalid property "recipient"
    * @dataProvider invalidRecipientDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidRecipient(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRecipient($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validRecipientDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_recipient'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidRecipientDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_recipient'));
    }

    /**
    * Test property "kbk"
    * @dataProvider validKbkDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testKbk(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getKbk());
        self::assertEmpty($instance->kbk);
        $instance->setKbk($value);
        self::assertEquals($value, $instance->getKbk());
        self::assertEquals($value, $instance->kbk);
        if (!empty($value)) {
            self::assertNotNull($instance->getKbk());
            self::assertNotNull($instance->kbk);
            self::assertMatchesRegularExpression("/^(\d{20})|(0)$/", $instance->getKbk());
            self::assertMatchesRegularExpression("/^(\d{20})|(0)$/", $instance->kbk);
            self::assertLessThanOrEqual(20, is_string($instance->getKbk()) ? mb_strlen($instance->getKbk()) : $instance->getKbk());
            self::assertLessThanOrEqual(20, is_string($instance->kbk) ? mb_strlen($instance->kbk) : $instance->kbk);
            self::assertGreaterThanOrEqual(1, is_string($instance->getKbk()) ? mb_strlen($instance->getKbk()) : $instance->getKbk());
            self::assertGreaterThanOrEqual(1, is_string($instance->kbk) ? mb_strlen($instance->kbk) : $instance->kbk);
        }
    }

    /**
    * Test invalid property "kbk"
    * @dataProvider invalidKbkDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidKbk(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setKbk($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validKbkDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_kbk'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidKbkDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_kbk'));
    }

    /**
    * Test property "oktmo"
    * @dataProvider validOktmoDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testOktmo(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getOktmo());
        self::assertEmpty($instance->oktmo);
        $instance->setOktmo($value);
        self::assertEquals($value, $instance->getOktmo());
        self::assertEquals($value, $instance->oktmo);
        if (!empty($value)) {
            self::assertNotNull($instance->getOktmo());
            self::assertNotNull($instance->oktmo);
            self::assertMatchesRegularExpression("/^(\d{8})|(0)$/", $instance->getOktmo());
            self::assertMatchesRegularExpression("/^(\d{8})|(0)$/", $instance->oktmo);
            self::assertLessThanOrEqual(8, is_string($instance->getOktmo()) ? mb_strlen($instance->getOktmo()) : $instance->getOktmo());
            self::assertLessThanOrEqual(8, is_string($instance->oktmo) ? mb_strlen($instance->oktmo) : $instance->oktmo);
            self::assertGreaterThanOrEqual(1, is_string($instance->getOktmo()) ? mb_strlen($instance->getOktmo()) : $instance->getOktmo());
            self::assertGreaterThanOrEqual(1, is_string($instance->oktmo) ? mb_strlen($instance->oktmo) : $instance->oktmo);
        }
    }

    /**
    * Test invalid property "oktmo"
    * @dataProvider invalidOktmoDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidOktmo(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setOktmo($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validOktmoDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_oktmo'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidOktmoDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_oktmo'));
    }

    /**
    * Test property "payment_period"
    * @dataProvider validPaymentPeriodDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testPaymentPeriod(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentPeriod());
        self::assertEmpty($instance->payment_period);
        $instance->setPaymentPeriod($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentPeriod()->toArray() : $instance->getPaymentPeriod());
        self::assertEquals($value, is_array($value) ? $instance->payment_period->toArray() : $instance->payment_period);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentPeriod());
            self::assertNotNull($instance->payment_period);
        }
    }

    /**
    * Test invalid property "payment_period"
    * @dataProvider invalidPaymentPeriodDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidPaymentPeriod(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentPeriod($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validPaymentPeriodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_period'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidPaymentPeriodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_period'));
    }

    /**
    * Test property "payment_document_id"
    * @dataProvider validPaymentDocumentIdDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testPaymentDocumentId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentDocumentId());
        self::assertEmpty($instance->payment_document_id);
        $instance->setPaymentDocumentId($value);
        self::assertEquals($value, $instance->getPaymentDocumentId());
        self::assertEquals($value, $instance->payment_document_id);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentDocumentId());
            self::assertNotNull($instance->payment_document_id);
            self::assertTrue($instance->hasPaymentDocumentId());
            self::assertTrue($instance->validate());
            self::assertLessThanOrEqual(18, is_string($instance->getPaymentDocumentId()) ? mb_strlen($instance->getPaymentDocumentId()) : $instance->getPaymentDocumentId());
            self::assertLessThanOrEqual(18, is_string($instance->payment_document_id) ? mb_strlen($instance->payment_document_id) : $instance->payment_document_id);
            self::assertGreaterThanOrEqual(18, is_string($instance->getPaymentDocumentId()) ? mb_strlen($instance->getPaymentDocumentId()) : $instance->getPaymentDocumentId());
            self::assertGreaterThanOrEqual(18, is_string($instance->payment_document_id) ? mb_strlen($instance->payment_document_id) : $instance->payment_document_id);
        }
    }

    /**
    * Test invalid property "payment_document_id"
    * @dataProvider invalidPaymentDocumentIdDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidPaymentDocumentId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentDocumentId($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validPaymentDocumentIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_document_id'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidPaymentDocumentIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_document_id'));
    }

    /**
    * Test property "payment_document_number"
    * @dataProvider validPaymentDocumentNumberDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testPaymentDocumentNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentDocumentNumber());
        self::assertEmpty($instance->payment_document_number);
        $instance->setPaymentDocumentNumber($value);
        self::assertEquals($value, $instance->getPaymentDocumentNumber());
        self::assertEquals($value, $instance->payment_document_number);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentDocumentNumber());
            self::assertNotNull($instance->payment_document_number);
            self::assertTrue($instance->hasPaymentDocumentNumber());
            self::assertTrue($instance->validate());
            self::assertMatchesRegularExpression("/^[0-9а-яА-Яa-zA-Z]{1,30}$/u", $instance->getPaymentDocumentNumber());
            self::assertMatchesRegularExpression("/^[0-9а-яА-Яa-zA-Z]{1,30}$/u", $instance->payment_document_number);
            self::assertLessThanOrEqual(30, is_string($instance->getPaymentDocumentNumber()) ? mb_strlen($instance->getPaymentDocumentNumber()) : $instance->getPaymentDocumentNumber());
            self::assertLessThanOrEqual(30, is_string($instance->payment_document_number) ? mb_strlen($instance->payment_document_number) : $instance->payment_document_number);
            self::assertGreaterThanOrEqual(1, is_string($instance->getPaymentDocumentNumber()) ? mb_strlen($instance->getPaymentDocumentNumber()) : $instance->getPaymentDocumentNumber());
            self::assertGreaterThanOrEqual(1, is_string($instance->payment_document_number) ? mb_strlen($instance->payment_document_number) : $instance->payment_document_number);
        }
    }

    /**
    * Test invalid property "payment_document_number"
    * @dataProvider invalidPaymentDocumentNumberDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidPaymentDocumentNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentDocumentNumber($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validPaymentDocumentNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_document_number'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidPaymentDocumentNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_document_number'));
    }

    /**
    * Test property "account_number"
    * @dataProvider validAccountNumberDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testAccountNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getAccountNumber());
        self::assertEmpty($instance->account_number);
        $instance->setAccountNumber($value);
        self::assertEquals($value, $instance->getAccountNumber());
        self::assertEquals($value, $instance->account_number);
        if (!empty($value)) {
            self::assertNotNull($instance->getAccountNumber());
            self::assertNotNull($instance->account_number);
            self::assertTrue($instance->hasAccountNumber());
            self::assertTrue($instance->validate());
            self::assertMatchesRegularExpression("/^[0-9а-яА-Яa-zA-Z]{1,30}$/u", $instance->getAccountNumber());
            self::assertMatchesRegularExpression("/^[0-9а-яА-Яa-zA-Z]{1,30}$/u", $instance->account_number);
            self::assertLessThanOrEqual(30, is_string($instance->getAccountNumber()) ? mb_strlen($instance->getAccountNumber()) : $instance->getAccountNumber());
            self::assertLessThanOrEqual(30, is_string($instance->account_number) ? mb_strlen($instance->account_number) : $instance->account_number);
            self::assertGreaterThanOrEqual(1, is_string($instance->getAccountNumber()) ? mb_strlen($instance->getAccountNumber()) : $instance->getAccountNumber());
            self::assertGreaterThanOrEqual(1, is_string($instance->account_number) ? mb_strlen($instance->account_number) : $instance->account_number);
        }
    }

    /**
    * Test invalid property "account_number"
    * @dataProvider invalidAccountNumberDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidAccountNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAccountNumber($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validAccountNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_account_number'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidAccountNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_account_number'));
    }

    /**
    * Test property "unified_account_number"
    * @dataProvider validUnifiedAccountNumberDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testUnifiedAccountNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getUnifiedAccountNumber());
        self::assertEmpty($instance->unified_account_number);
        $instance->setUnifiedAccountNumber($value);
        self::assertEquals($value, $instance->getUnifiedAccountNumber());
        self::assertEquals($value, $instance->unified_account_number);
        if (!empty($value)) {
            self::assertNotNull($instance->getUnifiedAccountNumber());
            self::assertNotNull($instance->unified_account_number);
            self::assertTrue($instance->hasUnifiedAccountNumber());
            self::assertTrue($instance->validate());
            self::assertLessThanOrEqual(10, is_string($instance->getUnifiedAccountNumber()) ? mb_strlen($instance->getUnifiedAccountNumber()) : $instance->getUnifiedAccountNumber());
            self::assertLessThanOrEqual(10, is_string($instance->unified_account_number) ? mb_strlen($instance->unified_account_number) : $instance->unified_account_number);
            self::assertGreaterThanOrEqual(10, is_string($instance->getUnifiedAccountNumber()) ? mb_strlen($instance->getUnifiedAccountNumber()) : $instance->getUnifiedAccountNumber());
            self::assertGreaterThanOrEqual(10, is_string($instance->unified_account_number) ? mb_strlen($instance->unified_account_number) : $instance->unified_account_number);
        }
    }

    /**
    * Test invalid property "unified_account_number"
    * @dataProvider invalidUnifiedAccountNumberDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidUnifiedAccountNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setUnifiedAccountNumber($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validUnifiedAccountNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_unified_account_number'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidUnifiedAccountNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_unified_account_number'));
    }

    /**
    * Test property "service_id"
    * @dataProvider validServiceIdDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testServiceId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getServiceId());
        self::assertEmpty($instance->service_id);
        $instance->setServiceId($value);
        self::assertEquals($value, $instance->getServiceId());
        self::assertEquals($value, $instance->service_id);
        if (!empty($value)) {
            self::assertNotNull($instance->getServiceId());
            self::assertNotNull($instance->service_id);
            self::assertTrue($instance->hasServiceId());
            self::assertTrue($instance->validate());
            self::assertLessThanOrEqual(13, is_string($instance->getServiceId()) ? mb_strlen($instance->getServiceId()) : $instance->getServiceId());
            self::assertLessThanOrEqual(13, is_string($instance->service_id) ? mb_strlen($instance->service_id) : $instance->service_id);
            self::assertGreaterThanOrEqual(13, is_string($instance->getServiceId()) ? mb_strlen($instance->getServiceId()) : $instance->getServiceId());
            self::assertGreaterThanOrEqual(13, is_string($instance->service_id) ? mb_strlen($instance->service_id) : $instance->service_id);
        }
    }

    /**
    * Test invalid property "service_id"
    * @dataProvider invalidServiceIdDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidServiceId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setServiceId($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validServiceIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_service_id'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidServiceIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_service_id'));
    }

    /**
     * @return void
     */
    public function testInvalidValidate(): void
    {
        $instance = $this->getTestInstance();
        self::assertFalse($instance->validate());
    }
}
