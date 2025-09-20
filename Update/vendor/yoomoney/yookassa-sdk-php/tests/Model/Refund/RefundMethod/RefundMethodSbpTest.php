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

namespace Tests\YooKassa\Model\Refund\RefundMethod;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Refund\RefundMethod\RefundMethodSbp;

/**
 * RefundMethodSbpTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class RefundMethodSbpTest extends AbstractTestCase
{
    protected RefundMethodSbp $object;

    /**
     * @return RefundMethodSbp
     */
    protected function getTestInstance(): RefundMethodSbp
    {
        return new RefundMethodSbp();
    }

    /**
     * @return void
     */
    public function testRefundMethodSbpClassExists(): void
    {
        $this->object = $this->getMockBuilder(RefundMethodSbp::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(RefundMethodSbp::class));
        $this->assertInstanceOf(RefundMethodSbp::class, $this->object);
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
        self::assertContains($instance->getType(), ['sbp']);
        self::assertContains($instance->type, ['sbp']);
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
     * Test property "sbp_operation_id"
     * @dataProvider validSbpOperationIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSbpOperationId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getSbpOperationId());
        self::assertEmpty($instance->sbp_operation_id);
        $instance->setSbpOperationId($value);
        self::assertEquals($value, $instance->getSbpOperationId());
        self::assertEquals($value, $instance->sbp_operation_id);
        if (!empty($value)) {
            self::assertNotNull($instance->getSbpOperationId());
            self::assertNotNull($instance->sbp_operation_id);
        }
    }

    /**
     * Test invalid property "sbp_operation_id"
     * @dataProvider invalidSbpOperationIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSbpOperationId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSbpOperationId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSbpOperationIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_sbp_operation_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSbpOperationIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_sbp_operation_id'));
    }
}
