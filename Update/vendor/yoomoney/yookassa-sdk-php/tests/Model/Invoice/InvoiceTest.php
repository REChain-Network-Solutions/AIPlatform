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

namespace Tests\YooKassa\Model\Invoice;

use Datetime;
use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Model\Invoice\Invoice;
use YooKassa\Model\Invoice\InvoiceInterface;
use YooKassa\Model\Metadata;

/**
 * InvoiceTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class InvoiceTest extends AbstractTestCase
{
    protected Invoice $object;

    /**
     * @return Invoice
     */
    protected function getTestInstance(): Invoice
    {
        return new Invoice();
    }

    /**
     * @return void
     */
    public function testInvoiceClassExists(): void
    {
        $this->object = $this->getMockBuilder(Invoice::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Invoice::class));
        $this->assertInstanceOf(Invoice::class, $this->object);
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
        self::assertEquals($value, $instance->getId());
        self::assertEquals($value, $instance->id);
        self::assertLessThanOrEqual(InvoiceInterface::MIN_LENGTH_ID, is_string($instance->getId()) ? mb_strlen($instance->getId()) : $instance->getId());
        self::assertLessThanOrEqual(InvoiceInterface::MIN_LENGTH_ID, is_string($instance->id) ? mb_strlen($instance->id) : $instance->id);
        self::assertGreaterThanOrEqual(InvoiceInterface::MAX_LENGTH_ID, is_string($instance->getId()) ? mb_strlen($instance->getId()) : $instance->getId());
        self::assertGreaterThanOrEqual(InvoiceInterface::MAX_LENGTH_ID, is_string($instance->id) ? mb_strlen($instance->id) : $instance->id);
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
        self::assertEquals($value, $instance->getStatus());
        self::assertEquals($value, $instance->status);
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
     * Test property "cart"
     * @dataProvider validCartDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCart(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertIsObject($instance->getCart());
        self::assertIsObject($instance->cart);
        self::assertCount(0, $instance->getCart());
        self::assertCount(0, $instance->cart);
        $instance->setCart($value);
        self::assertNotNull($instance->getCart());
        self::assertNotNull($instance->cart);
        foreach ($value as $key => $element) {
            if (is_array($element) && !empty($element)) {
                self::assertEquals($element, $instance->getCart()[$key]->toArray());
                self::assertEquals($element, $instance->cart[$key]->toArray());
                self::assertIsArray($instance->getCart()[$key]->toArray());
                self::assertIsArray($instance->cart[$key]->toArray());
            }
            if (is_object($element) && !empty($element)) {
                self::assertEquals($element, $instance->getCart()->get($key));
                self::assertIsObject($instance->getCart()->get($key));
                self::assertIsObject($instance->cart->get($key));
                self::assertIsObject($instance->getCart());
                self::assertIsObject($instance->cart);
            }
        }
        self::assertCount(count($value), $instance->getCart());
        self::assertCount(count($value), $instance->cart);
    }

    /**
     * Test invalid property "cart"
     * @dataProvider invalidCartDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCart(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCart($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCartDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_cart'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCartDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_cart'));
    }

    /**
     * Test property "delivery_method"
     * @dataProvider validDeliveryMethodDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDeliveryMethod(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getDeliveryMethod());
        self::assertEmpty($instance->delivery_method);
        $instance->setDeliveryMethod($value);
        self::assertEquals($value, is_array($value) ? $instance->getDeliveryMethod()->toArray() : $instance->getDeliveryMethod());
        self::assertEquals($value, is_array($value) ? $instance->delivery_method->toArray() : $instance->delivery_method);
        if (!empty($value)) {
            self::assertNotNull($instance->getDeliveryMethod());
            self::assertNotNull($instance->delivery_method);
        }
    }

    /**
     * Test invalid property "delivery_method"
     * @dataProvider invalidDeliveryMethodDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDeliveryMethod(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDeliveryMethod($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDeliveryMethodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_delivery_method'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDeliveryMethodDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_delivery_method'));
    }

    /**
     * Test property "payment_details"
     * @dataProvider validPaymentDetailsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPaymentDetails(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPaymentDetails());
        self::assertEmpty($instance->payment_details);
        $instance->setPaymentDetails($value);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentDetails()->toArray() : $instance->getPaymentDetails());
        self::assertEquals($value, is_array($value) ? $instance->payment_details->toArray() : $instance->payment_details);
        if (!empty($value)) {
            self::assertNotNull($instance->getPaymentDetails());
            self::assertNotNull($instance->payment_details);
        }
    }

    /**
     * Test invalid property "payment_details"
     * @dataProvider invalidPaymentDetailsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPaymentDetails(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentDetails($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPaymentDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_details'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPaymentDetailsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_details'));
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
        self::assertEquals($value, $instance->getDescription());
        self::assertEquals($value, $instance->description);
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
}
