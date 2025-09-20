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

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Model\Invoice\LineItem;

/**
 * LineItemTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class LineItemTest extends AbstractTestCase
{
    protected LineItem $object;

    /**
     * @return LineItem
     */
    protected function getTestInstance(): LineItem
    {
        return new LineItem();
    }

    /**
     * @return void
     */
    public function testLineItemClassExists(): void
    {
        $this->object = $this->getMockBuilder(LineItem::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(LineItem::class));
        $this->assertInstanceOf(LineItem::class, $this->object);
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
        $instance->setDescription($value);
        self::assertNotNull($instance->getDescription());
        self::assertNotNull($instance->description);
        self::assertEquals($value, $instance->getDescription());
        self::assertEquals($value, $instance->description);
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
     * Test property "price"
     * @dataProvider validPriceDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPrice(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPrice($value);
        self::assertNotNull($instance->getPrice());
        self::assertNotNull($instance->price);
        self::assertEquals($value, is_array($value) ? $instance->getPrice()->toArray() : $instance->getPrice());
        self::assertEquals($value, is_array($value) ? $instance->price->toArray() : $instance->price);
    }

    /**
     * Test invalid property "price"
     * @dataProvider invalidPriceDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPrice(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPrice($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPriceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_price'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPriceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_price'));
    }

    /**
     * Test property "discount_price"
     * @dataProvider validDiscountPriceDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDiscountPrice(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getDiscountPrice());
        self::assertEmpty($instance->discount_price);
        $instance->setDiscountPrice($value);
        self::assertEquals($value, is_array($value) ? $instance->getDiscountPrice()->toArray() : $instance->getDiscountPrice());
        self::assertEquals($value, is_array($value) ? $instance->discount_price->toArray() : $instance->discount_price);
        if (!empty($value)) {
            self::assertNotNull($instance->getDiscountPrice());
            self::assertNotNull($instance->discount_price);
        }
    }

    /**
     * Test invalid property "discount_price"
     * @dataProvider invalidDiscountPriceDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDiscountPrice(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDiscountPrice($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDiscountPriceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_discount_price'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDiscountPriceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_discount_price'));
    }

    /**
     * Test property "quantity"
     * @dataProvider validQuantityDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testQuantity(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setQuantity($value);
        self::assertNotNull($instance->getQuantity());
        self::assertNotNull($instance->quantity);
        self::assertEquals($value, $instance->getQuantity());
        self::assertEquals($value, $instance->quantity);
        self::assertIsNumeric($instance->getQuantity());
        self::assertIsNumeric($instance->quantity);
    }

    /**
     * Test invalid property "quantity"
     * @dataProvider invalidQuantityDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidQuantity(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setQuantity($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_quantity'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_quantity'));
    }
}
