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

namespace Tests\YooKassa\Model\Refund;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Refund\Refund;

/**
 * RefundTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class RefundTest extends AbstractTestCase
{
    protected Refund $object;

    /**
     * @return Refund
     */
    protected function getTestInstance(): Refund
    {
        return new Refund();
    }

    /**
     * @return void
     */
    public function testRefundClassExists(): void
    {
        $this->object = $this->getMockBuilder(Refund::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Refund::class));
        $this->assertInstanceOf(Refund::class, $this->object);
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
     * Test property "payment_id"
     * @dataProvider validPaymentIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPaymentId($value);
        self::assertNotNull($instance->getPaymentId());
        self::assertNotNull($instance->payment_id);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentId()->toArray() : $instance->getPaymentId());
        self::assertEquals($value, is_array($value) ? $instance->payment_id->toArray() : $instance->payment_id);
        self::assertLessThanOrEqual(36, is_string($instance->getPaymentId()) ? mb_strlen($instance->getPaymentId()) : $instance->getPaymentId());
        self::assertLessThanOrEqual(36, is_string($instance->payment_id) ? mb_strlen($instance->payment_id) : $instance->payment_id);
        self::assertGreaterThanOrEqual(36, is_string($instance->getPaymentId()) ? mb_strlen($instance->getPaymentId()) : $instance->getPaymentId());
        self::assertGreaterThanOrEqual(36, is_string($instance->payment_id) ? mb_strlen($instance->payment_id) : $instance->payment_id);
    }

    /**
     * Test invalid property "payment_id"
     * @dataProvider invalidPaymentIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_id'));
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
            self::assertLessThanOrEqual(250, is_string($instance->getDescription()) ? mb_strlen($instance->getDescription()) : $instance->getDescription());
            self::assertLessThanOrEqual(250, is_string($instance->description) ? mb_strlen($instance->description) : $instance->description);
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
     * Test property "sources"
     * @dataProvider validSourcesDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSources(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getSources());
        self::assertEmpty($instance->sources);
        self::assertIsObject($instance->getSources());
        self::assertIsObject($instance->sources);
        self::assertCount(0, $instance->getSources());
        self::assertCount(0, $instance->sources);
        $instance->setSources($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getSources());
            self::assertNotNull($instance->sources);
            foreach ($value as $key => $element) {
                if (is_array($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getSources()[$key]->toArray());
                    self::assertEquals($element, $instance->sources[$key]->toArray());
                    self::assertIsArray($instance->getSources()[$key]->toArray());
                    self::assertIsArray($instance->sources[$key]->toArray());
                }
                if (is_object($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getSources()->get($key));
                    self::assertIsObject($instance->getSources()->get($key));
                    self::assertIsObject($instance->sources->get($key));
                    self::assertIsObject($instance->getSources());
                    self::assertIsObject($instance->sources);
                }
            }
            self::assertCount(count($value), $instance->getSources());
            self::assertCount(count($value), $instance->sources);
        }
    }

    /**
     * Test invalid property "sources"
     * @dataProvider invalidSourcesDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSources(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSources($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSourcesDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_sources'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSourcesDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_sources'));
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
     * Test property "refund_method"
     * @dataProvider validRefundMethodDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testRefundMethod(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getRefundMethod());
        self::assertEmpty($instance->refund_method);
        $instance->setRefundMethod($value);
        self::assertEquals($value, is_array($value) ? $instance->getRefundMethod()->toArray() : $instance->getRefundMethod());
        self::assertEquals($value, is_array($value) ? $instance->refund_method->toArray() : $instance->refund_method);
        if (!empty($value)) {
            self::assertNotNull($instance->getRefundMethod());
            self::assertNotNull($instance->refund_method);
        }
    }

    /**
     * Test invalid property "refund_method"
     * @dataProvider invalidRefundMethodDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidRefundMethod(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRefundMethod($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validRefundMethodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_refund_method'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidRefundMethodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_refund_method'));
    }

    /**
     * Test property "refund_authorization_details"
     * @dataProvider validRefundAuthorizationDetailsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testRefundAuthorizationDetails(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getRefundAuthorizationDetails());
        self::assertEmpty($instance->refund_authorization_details);
        $instance->setRefundAuthorizationDetails($value);
        self::assertEquals($value, is_array($value) ? $instance->getRefundAuthorizationDetails()->toArray() : $instance->getRefundAuthorizationDetails());
        self::assertEquals($value, is_array($value) ? $instance->refund_authorization_details->toArray() : $instance->refund_authorization_details);
        if (!empty($value)) {
            self::assertNotNull($instance->getRefundAuthorizationDetails());
            self::assertNotNull($instance->refund_authorization_details);
        }
    }

    /**
     * Test invalid property "refund_authorization_details"
     * @dataProvider invalidRefundAuthorizationDetailsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidRefundAuthorizationDetails(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRefundAuthorizationDetails($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validRefundAuthorizationDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_refund_authorization_details'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidRefundAuthorizationDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_refund_authorization_details'));
    }
}
