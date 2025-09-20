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

namespace Tests\YooKassa\Model\Payment;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\Payment;
use YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException;

/**
 * PaymentTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentTest extends AbstractTestCase
{
    protected Payment $object;

    /**
     * @return Payment
     */
    protected function getTestInstance(): Payment
    {
        return new Payment();
    }

    /**
     * @return void
     */
    public function testPaymentClassExists(): void
    {
        $this->object = $this->getMockBuilder(Payment::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Payment::class));
        $this->assertInstanceOf(Payment::class, $this->object);
    }

    /**
     * Test property "id"
     * @dataProvider validIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setId($value);
        self::assertNotNull($instance->getId());
        self::assertNotNull($instance->id);
        self::assertEquals($value, is_array($value) ? $instance->getId()->toArray() : $instance->getId());
        self::assertEquals($value, is_array($value) ? $instance->id->toArray() : $instance->id);
        self::assertLessThanOrEqual(36, is_string($instance->getId()) ? mb_strlen($instance->getId()) : $instance->getId());
        self::assertLessThanOrEqual(36, is_string($instance->id) ? mb_strlen($instance->id) : $instance->id);
        self::assertGreaterThanOrEqual(36, is_string($instance->getId()) ? mb_strlen($instance->getId()) : $instance->getId());
        self::assertGreaterThanOrEqual(36, is_string($instance->id) ? mb_strlen($instance->id) : $instance->id);
    }

    /**
     * Test invalid property "id"
     * @dataProvider invalidIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_id'));
    }

    /**
     * Test property "status"
     * @dataProvider validStatusDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testStatus(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setStatus($value);
        self::assertNotNull($instance->getStatus());
        self::assertNotNull($instance->status);
        self::assertEquals($value, is_array($value) ? $instance->getStatus()->toArray() : $instance->getStatus());
        self::assertEquals($value, is_array($value) ? $instance->status->toArray() : $instance->status);
    }

    /**
     * Test invalid property "status"
     * @dataProvider invalidStatusDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidStatus(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setStatus($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
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
     * Test property "income_amount"
     * @dataProvider validIncomeAmountDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testIncomeAmount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getIncomeAmount());
        self::assertEmpty($instance->income_amount);
        $instance->setIncomeAmount($value);
        self::assertEquals($value, is_array($value) ? $instance->getIncomeAmount()->toArray() : $instance->getIncomeAmount());
        self::assertEquals($value, is_array($value) ? $instance->income_amount->toArray() : $instance->income_amount);
        if (!empty($value)) {
            self::assertNotNull($instance->getIncomeAmount());
            self::assertNotNull($instance->income_amount);
        }
    }

    /**
     * Test invalid property "income_amount"
     * @dataProvider invalidIncomeAmountDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidIncomeAmount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setIncomeAmount($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validIncomeAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_income_amount'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidIncomeAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_income_amount'));
    }

    /**
     * Test property "description"
     * @dataProvider validDescriptionDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDescription(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getDescription());
        self::assertEmpty($instance->description);
        $instance->setDescription($value);
        self::assertEquals($value, is_array($value) ? $instance->getDescription()->toArray() : $instance->getDescription());
        self::assertEquals($value, is_array($value) ? $instance->description->toArray() : $instance->description);
        if (!empty($value)) {
            self::assertNotNull($instance->getDescription());
            self::assertNotNull($instance->description);
            self::assertLessThanOrEqual(128, is_string($instance->getDescription()) ? mb_strlen($instance->getDescription()) : $instance->getDescription());
            self::assertLessThanOrEqual(128, is_string($instance->description) ? mb_strlen($instance->description) : $instance->description);
        }
    }

    /**
     * Test invalid property "description"
     * @dataProvider invalidDescriptionDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDescription(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDescription($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDescriptionDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_description'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDescriptionDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_description'));
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
     * Test property "payment_method"
     * @dataProvider validPaymentMethodDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentMethod(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentMethod());
        self::assertEmpty($instance->payment_method);
        $instance->setPaymentMethod($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentMethod()->toArray() : $instance->getPaymentMethod());
        self::assertEquals($value, is_array($value) ? $instance->payment_method->toArray() : $instance->payment_method);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentMethod());
            self::assertNotNull($instance->payment_method);
        }
    }

    /**
     * Test invalid property "payment_method"
     * @dataProvider invalidPaymentMethodDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentMethod(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentMethod($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentMethodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_method'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentMethodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_method'));
    }

    /**
     * Test property "captured_at"
     * @dataProvider validCapturedAtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCapturedAt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCapturedAt());
        self::assertEmpty($instance->captured_at);
        $instance->setCapturedAt($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getCapturedAt());
            self::assertNotNull($instance->captured_at);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCapturedAt());
                self::assertEquals($value, $instance->captured_at);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCapturedAt());
                self::assertEquals(new Datetime($value), $instance->captured_at);
            }
        }
    }

    /**
     * Test invalid property "captured_at"
     * @dataProvider invalidCapturedAtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCapturedAt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCapturedAt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCapturedAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCapturedAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at'));
    }

    /**
     * Test property "created_at"
     * @dataProvider validCreatedAtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setCreatedAt($value);
        self::assertNotNull($instance->getCreatedAt());
        self::assertNotNull($instance->created_at);
        if ($value instanceof Datetime) {
            self::assertEquals($value, $instance->getCreatedAt());
            self::assertEquals($value, $instance->created_at);
        } else {
            self::assertEquals(new Datetime($value), $instance->getCreatedAt());
            self::assertEquals(new Datetime($value), $instance->created_at);
        }
    }

    /**
     * Test invalid property "created_at"
     * @dataProvider invalidCreatedAtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at'));
    }

    /**
     * Test property "expires_at"
     * @dataProvider validExpiresAtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiresAt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getExpiresAt());
        self::assertEmpty($instance->expires_at);
        $instance->setExpiresAt($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getExpiresAt());
            self::assertNotNull($instance->expires_at);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getExpiresAt());
                self::assertEquals($value, $instance->expires_at);
            } else {
                self::assertEquals(new Datetime($value), $instance->getExpiresAt());
                self::assertEquals(new Datetime($value), $instance->expires_at);
            }
        }
    }

    /**
     * Test invalid property "expires_at"
     * @dataProvider invalidExpiresAtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiresAt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiresAt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiresAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiresAtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expires_at'));
    }

    /**
     * Test property "confirmation"
     * @dataProvider validConfirmationDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testConfirmation(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getConfirmation());
        self::assertEmpty($instance->confirmation);
        $instance->setConfirmation($value);
        self::assertEquals($value, is_array($value) ? $instance->getConfirmation()->toArray() : $instance->getConfirmation());
        self::assertEquals($value, is_array($value) ? $instance->confirmation->toArray() : $instance->confirmation);
        if (!empty($value)) {
            self::assertNotNull($instance->getConfirmation());
            self::assertNotNull($instance->confirmation);
        }
    }

    /**
     * Test invalid property "confirmation"
     * @dataProvider invalidConfirmationDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidConfirmation(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setConfirmation($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validConfirmationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidConfirmationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation'));
    }

    /**
     * Test property "test"
     * @dataProvider validTestDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testTest(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setTest($value);
        self::assertNotNull($instance->getTest());
        self::assertNotNull($instance->test);
        self::assertEquals($value, is_array($value) ? $instance->getTest()->toArray() : $instance->getTest());
        self::assertEquals($value, is_array($value) ? $instance->test->toArray() : $instance->test);
        self::assertIsBool($instance->getTest());
        self::assertIsBool($instance->test);
    }

    /**
     * Test invalid property "test"
     * @dataProvider invalidTestDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidTest(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException(InvalidPropertyValueTypeException::class);
        $instance->setTest($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validTestDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_test'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTestDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_test'));
    }

    /**
     * Test property "refunded_amount"
     * @dataProvider validRefundedAmountDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testRefundedAmount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getRefundedAmount());
        self::assertEmpty($instance->refunded_amount);
        $instance->setRefundedAmount($value);
        self::assertEquals($value, is_array($value) ? $instance->getRefundedAmount()->toArray() : $instance->getRefundedAmount());
        self::assertEquals($value, is_array($value) ? $instance->refunded_amount->toArray() : $instance->refunded_amount);
        if (!empty($value)) {
            self::assertNotNull($instance->getRefundedAmount());
            self::assertNotNull($instance->refunded_amount);
        }
    }

    /**
     * Test invalid property "refunded_amount"
     * @dataProvider invalidRefundedAmountDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidRefundedAmount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRefundedAmount($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validRefundedAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_refunded_amount'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidRefundedAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_refunded_amount'));
    }

    /**
     * Test property "paid"
     * @dataProvider validPaidDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaid(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPaid($value);
        self::assertNotNull($instance->getPaid());
        self::assertNotNull($instance->paid);
        self::assertEquals($value, is_array($value) ? $instance->getPaid()->toArray() : $instance->getPaid());
        self::assertEquals($value, is_array($value) ? $instance->paid->toArray() : $instance->paid);
        self::assertIsBool($instance->getPaid());
        self::assertIsBool($instance->paid);
    }

    /**
     * Test invalid property "paid"
     * @dataProvider invalidPaidDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaid(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaid($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaidDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_paid'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaidDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_paid'));
    }

    /**
     * Test property "refundable"
     * @dataProvider validRefundableDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testRefundable(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setRefundable($value);
        self::assertNotNull($instance->getRefundable());
        self::assertNotNull($instance->refundable);
        self::assertEquals($value, is_array($value) ? $instance->getRefundable()->toArray() : $instance->getRefundable());
        self::assertEquals($value, is_array($value) ? $instance->refundable->toArray() : $instance->refundable);
        self::assertIsBool($instance->getRefundable());
        self::assertIsBool($instance->refundable);
    }

    /**
     * Test invalid property "refundable"
     * @dataProvider invalidRefundableDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidRefundable(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRefundable($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validRefundableDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_refundable'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidRefundableDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_refundable'));
    }

    /**
     * Test property "receipt_registration"
     * @dataProvider validReceiptRegistrationDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testReceiptRegistration(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getReceiptRegistration());
        self::assertEmpty($instance->receipt_registration);
        $instance->setReceiptRegistration($value);
        self::assertEquals($value, is_array($value) ? $instance->getReceiptRegistration()->toArray() : $instance->getReceiptRegistration());
        self::assertEquals($value, is_array($value) ? $instance->receipt_registration->toArray() : $instance->receipt_registration);
        if (!empty($value)) {
            self::assertNotNull($instance->getReceiptRegistration());
            self::assertNotNull($instance->receipt_registration);
        }
    }

    /**
     * Test invalid property "receipt_registration"
     * @dataProvider invalidReceiptRegistrationDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidReceiptRegistration(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setReceiptRegistration($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validReceiptRegistrationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_receipt_registration'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidReceiptRegistrationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_receipt_registration'));
    }

    /**
     * Test property "metadata"
     * @dataProvider validMetadataDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMetadata(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMetadata());
        self::assertEmpty($instance->metadata);
        $instance->setMetadata($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getMetadata());
            self::assertNotNull($instance->metadata);
            foreach ($value as $key => $element) {
                if (!empty($element)) {
                    self::assertEquals($element, $instance->getMetadata()[$key]);
                    self::assertEquals($element, $instance->metadata[$key]);
                    self::assertIsObject($instance->getMetadata());
                    self::assertIsObject($instance->metadata);
                }
            }
            self::assertCount(count($value), $instance->getMetadata());
            self::assertCount(count($value), $instance->metadata);
            if ($instance->getMetadata() instanceof Metadata) {
                self::assertEquals($value, $instance->getMetadata()->toArray());
                self::assertEquals($value, $instance->metadata->toArray());
                self::assertCount(count($value), $instance->getMetadata());
                self::assertCount(count($value), $instance->metadata);
            }
        }
    }

    /**
     * Test invalid property "metadata"
     * @dataProvider invalidMetadataDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMetadata(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMetadata($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMetadataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_metadata'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMetadataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_metadata'));
    }

    /**
     * Test property "cancellation_details"
     * @dataProvider validCancellationDetailsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCancellationDetails(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCancellationDetails());
        self::assertEmpty($instance->cancellation_details);
        $instance->setCancellationDetails($value);
        self::assertEquals($value, is_array($value) ? $instance->getCancellationDetails()->toArray() : $instance->getCancellationDetails());
        self::assertEquals($value, is_array($value) ? $instance->cancellation_details->toArray() : $instance->cancellation_details);
        if (!empty($value)) {
            self::assertNotNull($instance->getCancellationDetails());
            self::assertNotNull($instance->cancellation_details);
        }
    }

    /**
     * Test invalid property "cancellation_details"
     * @dataProvider invalidCancellationDetailsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCancellationDetails(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCancellationDetails($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCancellationDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_cancellation_details'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCancellationDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_cancellation_details'));
    }

    /**
     * Test property "authorization_details"
     * @dataProvider validAuthorizationDetailsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAuthorizationDetails(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getAuthorizationDetails());
        self::assertEmpty($instance->authorization_details);
        $instance->setAuthorizationDetails($value);
        self::assertEquals($value, is_array($value) ? $instance->getAuthorizationDetails()->toArray() : $instance->getAuthorizationDetails());
        self::assertEquals($value, is_array($value) ? $instance->authorization_details->toArray() : $instance->authorization_details);
        if (!empty($value)) {
            self::assertNotNull($instance->getAuthorizationDetails());
            self::assertNotNull($instance->authorization_details);
        }
    }

    /**
     * Test invalid property "authorization_details"
     * @dataProvider invalidAuthorizationDetailsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAuthorizationDetails(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAuthorizationDetails($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAuthorizationDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_authorization_details'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAuthorizationDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_authorization_details'));
    }

    /**
     * Test property "transfers"
     * @dataProvider validTransfersDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testTransfers(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getTransfers());
        self::assertEmpty($instance->transfers);
        self::assertIsObject($instance->getTransfers());
        self::assertIsObject($instance->transfers);
        self::assertCount(0, $instance->getTransfers());
        self::assertCount(0, $instance->transfers);
        $instance->setTransfers($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getTransfers());
            self::assertNotNull($instance->transfers);
            foreach ($value as $key => $element) {
                if (is_array($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getTransfers()[$key]->toArray());
                    self::assertEquals($element, $instance->transfers[$key]->toArray());
                    self::assertIsArray($instance->getTransfers()[$key]->toArray());
                    self::assertIsArray($instance->transfers[$key]->toArray());
                }
                if (is_object($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getTransfers()->get($key));
                    self::assertIsObject($instance->getTransfers()->get($key));
                    self::assertIsObject($instance->transfers->get($key));
                    self::assertIsObject($instance->getTransfers());
                    self::assertIsObject($instance->transfers);
                }
            }
            self::assertCount(count($value), $instance->getTransfers());
            self::assertCount(count($value), $instance->transfers);
        }
    }

    /**
     * Test invalid property "transfers"
     * @dataProvider invalidTransfersDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidTransfers(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setTransfers($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validTransfersDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_transfers'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTransfersDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_transfers'));
    }

    /**
     * Test property "deal"
     * @dataProvider validDealDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDeal(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getDeal());
        self::assertEmpty($instance->deal);
        $instance->setDeal($value);
        self::assertEquals($value, is_array($value) ? $instance->getDeal()->toArray() : $instance->getDeal());
        self::assertEquals($value, is_array($value) ? $instance->deal->toArray() : $instance->deal);
        if (!empty($value)) {
            self::assertNotNull($instance->getDeal());
            self::assertNotNull($instance->deal);
        }
    }

    /**
     * Test invalid property "deal"
     * @dataProvider invalidDealDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDeal(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDeal($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDealDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_deal'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDealDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_deal'));
    }

    /**
     * Test property "merchant_customer_id"
     * @dataProvider validMerchantCustomerIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMerchantCustomerId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMerchantCustomerId());
        self::assertEmpty($instance->merchant_customer_id);
        $instance->setMerchantCustomerId($value);
        self::assertEquals($value, is_array($value) ? $instance->getMerchantCustomerId()->toArray() : $instance->getMerchantCustomerId());
        self::assertEquals($value, is_array($value) ? $instance->merchant_customer_id->toArray() : $instance->merchant_customer_id);
        if (!empty($value)) {
            self::assertNotNull($instance->getMerchantCustomerId());
            self::assertNotNull($instance->merchant_customer_id);
        }
    }

    /**
     * Test invalid property "merchant_customer_id"
     * @dataProvider invalidMerchantCustomerIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMerchantCustomerId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMerchantCustomerId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMerchantCustomerIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_merchant_customer_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMerchantCustomerIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_merchant_customer_id'));
    }
}
