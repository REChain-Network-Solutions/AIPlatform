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

namespace Tests\YooKassa\Model\Payout;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payout\SbpParticipantBank;

/**
 * SbpParticipantBankTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class SbpParticipantBankTest extends AbstractTestCase
{
    protected SbpParticipantBank $object;

    /**
     * @return SbpParticipantBank
     */
    protected function getTestInstance(): SbpParticipantBank
    {
        return new SbpParticipantBank();
    }

    /**
     * @return void
     */
    public function testSbpParticipantBankClassExists(): void
    {
        $this->object = $this->getMockBuilder(SbpParticipantBank::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(SbpParticipantBank::class));
        $this->assertInstanceOf(SbpParticipantBank::class, $this->object);
    }

    /**
     * Test property "bank_id"
     * @dataProvider validBankIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testBankId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBankId($value);
        self::assertNotNull($instance->getBankId());
        self::assertNotNull($instance->bank_id);
        self::assertEquals($value, is_array($value) ? $instance->getBankId()->toArray() : $instance->getBankId());
        self::assertEquals($value, is_array($value) ? $instance->bank_id->toArray() : $instance->bank_id);
        self::assertLessThanOrEqual(12, is_string($instance->getBankId()) ? mb_strlen($instance->getBankId()) : $instance->getBankId());
        self::assertLessThanOrEqual(12, is_string($instance->bank_id) ? mb_strlen($instance->bank_id) : $instance->bank_id);
    }

    /**
     * Test invalid property "bank_id"
     * @dataProvider invalidBankIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidBankId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBankId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validBankIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidBankIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_id'));
    }

    /**
     * Test property "name"
     * @dataProvider validNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setName($value);
        self::assertNotNull($instance->getName());
        self::assertNotNull($instance->name);
        self::assertEquals($value, is_array($value) ? $instance->getName()->toArray() : $instance->getName());
        self::assertEquals($value, is_array($value) ? $instance->name->toArray() : $instance->name);
        self::assertLessThanOrEqual(128, is_string($instance->getName()) ? mb_strlen($instance->getName()) : $instance->getName());
        self::assertLessThanOrEqual(128, is_string($instance->name) ? mb_strlen($instance->name) : $instance->name);
    }

    /**
     * Test invalid property "name"
     * @dataProvider invalidNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_name'));
    }

    /**
     * Test property "bic"
     * @dataProvider validBicDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testBic(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBic($value);
        self::assertNotNull($instance->getBic());
        self::assertNotNull($instance->bic);
        self::assertEquals($value, is_array($value) ? $instance->getBic()->toArray() : $instance->getBic());
        self::assertEquals($value, is_array($value) ? $instance->bic->toArray() : $instance->bic);
        self::assertMatchesRegularExpression("/\\d{9}/", $instance->getBic());
        self::assertMatchesRegularExpression("/\\d{9}/", $instance->bic);
    }

    /**
     * Test invalid property "bic"
     * @dataProvider invalidBicDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidBic(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBic($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validBicDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_bic'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidBicDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_bic'));
    }
}
