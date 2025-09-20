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

namespace Tests\YooKassa\Model\Receipt;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Receipt\OperationalDetails;

/**
 * OperationalDetailsTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class OperationalDetailsTest extends AbstractTestCase
{
    protected OperationalDetails $object;

    /**
     * @return OperationalDetails
     */
    protected function getTestInstance(): OperationalDetails
    {
        return new OperationalDetails();
    }

    /**
     * @return void
     */
    public function testOperationalDetailsClassExists(): void
    {
        $this->object = $this->getMockBuilder(OperationalDetails::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(OperationalDetails::class));
        $this->assertInstanceOf(OperationalDetails::class, $this->object);
    }

    /**
     * Test property "operation_id"
     * @dataProvider validOperationIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testOperationId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setOperationId($value);
        self::assertNotNull($instance->getOperationId());
        self::assertNotNull($instance->operation_id);
        self::assertEquals($value, is_array($value) ? $instance->getOperationId()->toArray() : $instance->getOperationId());
        self::assertEquals($value, is_array($value) ? $instance->operation_id->toArray() : $instance->operation_id);
        self::assertLessThanOrEqual(255, is_string($instance->getOperationId()) ? mb_strlen($instance->getOperationId()) : $instance->getOperationId());
        self::assertLessThanOrEqual(255, is_string($instance->operation_id) ? mb_strlen($instance->operation_id) : $instance->operation_id);
        self::assertGreaterThanOrEqual(0, is_string($instance->getOperationId()) ? mb_strlen($instance->getOperationId()) : $instance->getOperationId());
        self::assertGreaterThanOrEqual(0, is_string($instance->operation_id) ? mb_strlen($instance->operation_id) : $instance->operation_id);
        self::assertIsNumeric($instance->getOperationId());
        self::assertIsNumeric($instance->operation_id);
    }

    /**
     * Test invalid property "operation_id"
     * @dataProvider invalidOperationIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidOperationId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setOperationId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validOperationIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_operation_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidOperationIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_operation_id'));
    }

    /**
     * Test property "value"
     * @dataProvider validValueDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testValue(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setValue($value);
        self::assertNotNull($instance->getValue());
        self::assertNotNull($instance->value);
        self::assertEquals($value, is_array($value) ? $instance->getValue()->toArray() : $instance->getValue());
        self::assertEquals($value, is_array($value) ? $instance->value->toArray() : $instance->value);
        self::assertLessThanOrEqual(64, is_string($instance->getValue()) ? mb_strlen($instance->getValue()) : $instance->getValue());
        self::assertLessThanOrEqual(64, is_string($instance->value) ? mb_strlen($instance->value) : $instance->value);
    }

    /**
     * Test invalid property "value"
     * @dataProvider invalidValueDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidValue(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setValue($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validValueDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_value'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidValueDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_value'));
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
}
