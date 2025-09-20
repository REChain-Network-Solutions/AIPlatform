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

namespace Tests\YooKassa\Model\Payment\Confirmation;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\Confirmation\ConfirmationQr;

/**
 * ConfirmationQrTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ConfirmationQrTest extends AbstractTestCase
{
    protected ConfirmationQr $object;

    /**
     * @return ConfirmationQr
     */
    protected function getTestInstance(): ConfirmationQr
    {
        return new ConfirmationQr();
    }

    /**
     * @return void
     */
    public function testConfirmationQrClassExists(): void
    {
        $this->object = $this->getMockBuilder(ConfirmationQr::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ConfirmationQr::class));
        $this->assertInstanceOf(ConfirmationQr::class, $this->object);
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
        self::assertContains($instance->getType(), ['qr']);
        self::assertContains($instance->type, ['qr']);
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
     * Test property "confirmation_data"
     * @dataProvider validConfirmationDataDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testConfirmationData(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setConfirmationData($value);
        self::assertNotNull($instance->getConfirmationData());
        self::assertNotNull($instance->confirmation_data);
        self::assertEquals($value, is_array($value) ? $instance->getConfirmationData()->toArray() : $instance->getConfirmationData());
        self::assertEquals($value, is_array($value) ? $instance->confirmation_data->toArray() : $instance->confirmation_data);
    }

    /**
     * Test invalid property "confirmation_data"
     * @dataProvider invalidConfirmationDataDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidConfirmationData(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setConfirmationData($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validConfirmationDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation_data'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidConfirmationDataDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation_data'));
    }
}
