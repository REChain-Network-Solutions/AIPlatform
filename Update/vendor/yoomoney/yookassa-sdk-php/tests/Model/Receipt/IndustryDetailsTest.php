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
use YooKassa\Model\Receipt\IndustryDetails;

/**
 * IndustryDetailsTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class IndustryDetailsTest extends AbstractTestCase
{
    protected IndustryDetails $object;

    /**
     * @return IndustryDetails
     */
    protected function getTestInstance(): IndustryDetails
    {
        return new IndustryDetails();
    }

    /**
     * @return void
     */
    public function testIndustryDetailsClassExists(): void
    {
        $this->object = $this->getMockBuilder(IndustryDetails::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(IndustryDetails::class));
        $this->assertInstanceOf(IndustryDetails::class, $this->object);
    }

    /**
     * Test property "federal_id"
     * @dataProvider validFederalIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testFederalId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setFederalId($value);
        self::assertNotNull($instance->getFederalId());
        self::assertNotNull($instance->federal_id);
        self::assertEquals($value, is_array($value) ? $instance->getFederalId()->toArray() : $instance->getFederalId());
        self::assertEquals($value, is_array($value) ? $instance->federal_id->toArray() : $instance->federal_id);
        self::assertMatchesRegularExpression("/(^00[1-9]{1}$)|(^0[1-6]{1}[0-9]{1}$)|(^07[0-3]{1}$)/", $instance->getFederalId());
        self::assertMatchesRegularExpression("/(^00[1-9]{1}$)|(^0[1-6]{1}[0-9]{1}$)|(^07[0-3]{1}$)/", $instance->federal_id);
    }

    /**
     * Test invalid property "federal_id"
     * @dataProvider invalidFederalIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidFederalId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setFederalId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validFederalIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_federal_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidFederalIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_federal_id'));
    }

    /**
     * Test property "document_date"
     * @dataProvider validDocumentDateDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDocumentDate(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setDocumentDate($value);
        self::assertNotNull($instance->getDocumentDate());
        self::assertNotNull($instance->document_date);
        if ($value instanceof Datetime) {
            self::assertEquals($value, $instance->getDocumentDate());
            self::assertEquals($value, $instance->document_date);
        } else {
            self::assertEquals(new Datetime($value), $instance->getDocumentDate());
            self::assertEquals(new Datetime($value), $instance->document_date);
        }
    }

    /**
     * Test invalid property "document_date"
     * @dataProvider invalidDocumentDateDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDocumentDate(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDocumentDate($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDocumentDateDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_document_date'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDocumentDateDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_document_date'));
    }

    /**
     * Test property "document_number"
     * @dataProvider validDocumentNumberDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDocumentNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setDocumentNumber($value);
        self::assertNotNull($instance->getDocumentNumber());
        self::assertNotNull($instance->document_number);
        self::assertEquals($value, is_array($value) ? $instance->getDocumentNumber()->toArray() : $instance->getDocumentNumber());
        self::assertEquals($value, is_array($value) ? $instance->document_number->toArray() : $instance->document_number);
        self::assertLessThanOrEqual(32, is_string($instance->getDocumentNumber()) ? mb_strlen($instance->getDocumentNumber()) : $instance->getDocumentNumber());
        self::assertLessThanOrEqual(32, is_string($instance->document_number) ? mb_strlen($instance->document_number) : $instance->document_number);
    }

    /**
     * Test invalid property "document_number"
     * @dataProvider invalidDocumentNumberDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDocumentNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDocumentNumber($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDocumentNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_document_number'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDocumentNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_document_number'));
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
        self::assertLessThanOrEqual(256, is_string($instance->getValue()) ? mb_strlen($instance->getValue()) : $instance->getValue());
        self::assertLessThanOrEqual(256, is_string($instance->value) ? mb_strlen($instance->value) : $instance->value);
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
}
