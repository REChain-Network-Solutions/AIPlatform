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

namespace Tests\YooKassa\Request\Payments;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\PaymentsRequest;

/**
 * PaymentsRequestTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentsRequestTest extends AbstractTestCase
{
    protected PaymentsRequest $object;

    /**
     * @param mixed|null $value
     * @return PaymentsRequest
     */
    protected function getTestInstance(mixed $value = null): PaymentsRequest
    {
        return new PaymentsRequest($value);
    }

    /**
     * @return void
     */
    public function testPaymentsRequestClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentsRequest::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentsRequest::class));
        $this->assertInstanceOf(PaymentsRequest::class, $this->object);
        self::assertTrue($this->object->validate());
    }

    /**
     * Test property "cursor"
     * @dataProvider validCursorDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCursor(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCursor());
        self::assertEmpty($instance->cursor);
        $instance->setCursor($value);
        self::assertEquals($value, is_array($value) ? $instance->getCursor()->toArray() : $instance->getCursor());
        self::assertEquals($value, is_array($value) ? $instance->cursor->toArray() : $instance->cursor);
        if (!empty($value)) {
            self::assertTrue($instance->hasCursor());
            self::assertNotNull($instance->getCursor());
            self::assertNotNull($instance->cursor);
        }
    }

    /**
     * Test invalid property "cursor"
     * @dataProvider invalidCursorDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCursor(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCursor($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCursorDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_cursor'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCursorDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_cursor'));
    }

    /**
     * Test property "limit"
     * @dataProvider validLimitDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testLimit(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setLimit($value);
        self::assertEquals($value, is_array($value) ? $instance->getLimit()->toArray() : $instance->getLimit());
        self::assertEquals($value, is_array($value) ? $instance->limit->toArray() : $instance->limit);
        if (!empty($value)) {
            self::assertTrue($instance->hasLimit());
            self::assertNotNull($instance->getLimit());
            self::assertNotNull($instance->limit);
            self::assertLessThanOrEqual(100, is_string($instance->getLimit()) ? mb_strlen($instance->getLimit()) : $instance->getLimit());
            self::assertLessThanOrEqual(100, is_string($instance->limit) ? mb_strlen($instance->limit) : $instance->limit);
            self::assertGreaterThanOrEqual(1, is_string($instance->getLimit()) ? mb_strlen($instance->getLimit()) : $instance->getLimit());
            self::assertGreaterThanOrEqual(1, is_string($instance->limit) ? mb_strlen($instance->limit) : $instance->limit);
            self::assertIsNumeric($instance->getLimit());
            self::assertIsNumeric($instance->limit);
        }
    }

    /**
     * Test invalid property "limit"
     * @dataProvider invalidCursorDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidLimit(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCursor($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validLimitDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_limit'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidLimitDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_limit'));
    }

    /**
     * Test property "created_at_gte"
     * @dataProvider validCreatedAtGteDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAtGte(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCreatedAtGte());
        self::assertEmpty($instance->created_at_gte);
        $instance->setCreatedAtGte($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCreatedAtGte());
            self::assertNotNull($instance->getCreatedAtGte());
            self::assertNotNull($instance->created_at_gte);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCreatedAtGte());
                self::assertEquals($value, $instance->created_at_gte);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCreatedAtGte());
                self::assertEquals(new Datetime($value), $instance->created_at_gte);
            }
        }
    }

    /**
     * Test invalid property "created_at_gte"
     * @dataProvider invalidCreatedAtGteDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAtGte(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAtGte($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtGteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_gte'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtGteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_gte'));
    }

    /**
     * Test property "created_at_gt"
     * @dataProvider validCreatedAtGtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAtGt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCreatedAtGt());
        self::assertEmpty($instance->created_at_gt);
        $instance->setCreatedAtGt($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCreatedAtGt());
            self::assertNotNull($instance->getCreatedAtGt());
            self::assertNotNull($instance->created_at_gt);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCreatedAtGt());
                self::assertEquals($value, $instance->created_at_gt);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCreatedAtGt());
                self::assertEquals(new Datetime($value), $instance->created_at_gt);
            }
        }
    }

    /**
     * Test invalid property "created_at_gt"
     * @dataProvider invalidCreatedAtGtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAtGt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAtGt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtGtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_gt'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtGtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_gt'));
    }

    /**
     * Test property "created_at_lte"
     * @dataProvider validCreatedAtLteDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAtLte(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCreatedAtLte());
        self::assertEmpty($instance->created_at_lte);
        $instance->setCreatedAtLte($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCreatedAtLte());
            self::assertNotNull($instance->getCreatedAtLte());
            self::assertNotNull($instance->created_at_lte);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCreatedAtLte());
                self::assertEquals($value, $instance->created_at_lte);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCreatedAtLte());
                self::assertEquals(new Datetime($value), $instance->created_at_lte);
            }
        }
    }

    /**
     * Test invalid property "created_at_lte"
     * @dataProvider invalidCreatedAtLteDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAtLte(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAtLte($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtLteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_lte'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtLteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_lte'));
    }

    /**
     * Test property "created_at_lt"
     * @dataProvider validCreatedAtLtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCreatedAtLt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCreatedAtLt());
        self::assertEmpty($instance->created_at_lt);
        $instance->setCreatedAtLt($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCreatedAtLt());
            self::assertNotNull($instance->getCreatedAtLt());
            self::assertNotNull($instance->created_at_lt);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCreatedAtLt());
                self::assertEquals($value, $instance->created_at_lt);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCreatedAtLt());
                self::assertEquals(new Datetime($value), $instance->created_at_lt);
            }
        }
    }

    /**
     * Test invalid property "created_at_lt"
     * @dataProvider invalidCreatedAtLtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCreatedAtLt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCreatedAtLt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCreatedAtLtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_lt'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCreatedAtLtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_created_at_lt'));
    }

    /**
     * Test property "captured_at_gte"
     * @dataProvider validCapturedAtGteDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCapturedAtGte(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCapturedAtGte());
        self::assertEmpty($instance->captured_at_gte);
        $instance->setCapturedAtGte($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCapturedAtGte());
            self::assertNotNull($instance->getCapturedAtGte());
            self::assertNotNull($instance->captured_at_gte);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCapturedAtGte());
                self::assertEquals($value, $instance->captured_at_gte);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCapturedAtGte());
                self::assertEquals(new Datetime($value), $instance->captured_at_gte);
            }
        }
    }

    /**
     * Test invalid property "captured_at_gte"
     * @dataProvider invalidCapturedAtGteDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCapturedAtGte(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCapturedAtGte($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCapturedAtGteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at_gte'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCapturedAtGteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at_gte'));
    }

    /**
     * Test property "captured_at_gt"
     * @dataProvider validCapturedAtGtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCapturedAtGt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCapturedAtGt());
        self::assertEmpty($instance->captured_at_gt);
        $instance->setCapturedAtGt($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCapturedAtGt());
            self::assertNotNull($instance->getCapturedAtGt());
            self::assertNotNull($instance->captured_at_gt);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCapturedAtGt());
                self::assertEquals($value, $instance->captured_at_gt);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCapturedAtGt());
                self::assertEquals(new Datetime($value), $instance->captured_at_gt);
            }
        }
    }

    /**
     * Test invalid property "captured_at_gt"
     * @dataProvider invalidCapturedAtGtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCapturedAtGt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCapturedAtGt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCapturedAtGtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at_gt'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCapturedAtGtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at_gt'));
    }

    /**
     * Test property "captured_at_lte"
     * @dataProvider validCapturedAtLteDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCapturedAtLte(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCapturedAtLte());
        self::assertEmpty($instance->captured_at_lte);
        $instance->setCapturedAtLte($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCapturedAtLte());
            self::assertNotNull($instance->getCapturedAtLte());
            self::assertNotNull($instance->captured_at_lte);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCapturedAtLte());
                self::assertEquals($value, $instance->captured_at_lte);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCapturedAtLte());
                self::assertEquals(new Datetime($value), $instance->captured_at_lte);
            }
        }
    }

    /**
     * Test invalid property "captured_at_lte"
     * @dataProvider invalidCapturedAtLteDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCapturedAtLte(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCapturedAtLte($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCapturedAtLteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at_lte'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCapturedAtLteDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at_lte'));
    }

    /**
     * Test property "captured_at_lt"
     * @dataProvider validCapturedAtLtDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCapturedAtLt(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCapturedAtLt());
        self::assertEmpty($instance->captured_at_lt);
        $instance->setCapturedAtLt($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasCapturedAtLt());
            self::assertNotNull($instance->getCapturedAtLt());
            self::assertNotNull($instance->captured_at_lt);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getCapturedAtLt());
                self::assertEquals($value, $instance->captured_at_lt);
            } else {
                self::assertEquals(new Datetime($value), $instance->getCapturedAtLt());
                self::assertEquals(new Datetime($value), $instance->captured_at_lt);
            }
        }
    }

    /**
     * Test invalid property "captured_at_lt"
     * @dataProvider invalidCapturedAtLtDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCapturedAtLt(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCapturedAtLt($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCapturedAtLtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at_lt'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCapturedAtLtDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_captured_at_lt'));
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
            self::assertTrue($instance->hasPaymentMethod());
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
        self::assertEmpty($instance->getStatus());
        self::assertEmpty($instance->status);
        $instance->setStatus($value);
        if (!empty($value)) {
            self::assertTrue($instance->hasStatus());
            self::assertNotNull($instance->getStatus());
            self::assertNotNull($instance->status);
            self::assertEquals($value, is_array($value) ? $instance->getStatus()->toArray() : $instance->getStatus());
            self::assertEquals($value, is_array($value) ? $instance->status->toArray() : $instance->status);
            self::assertContains($instance->getStatus(), ['pending', 'waiting_for_capture', 'succeeded', 'canceled']);
            self::assertContains($instance->status, ['pending', 'waiting_for_capture', 'succeeded', 'canceled']);
        }
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
     * Test valid method "builder"
     * @dataProvider validClassDataProvider
     * @param mixed $value
     *
     * @return void
     */
    public function testBuilder(mixed $value): void
    {
        $instance = $this->getTestInstance($value);
        self::assertInstanceOf(AbstractRequestBuilder::class, $instance::builder());
    }

    /**
     * @return array
     * @throws Exception
     */
    public function validClassDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return [$this->getValidDataProviderByClass($instance)];
    }
}
