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
use YooKassa\Model\Payout\PayoutDestinationSbp;

/**
 * PayoutDestinationSbpTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PayoutDestinationSbpTest extends AbstractTestCase
{
    protected PayoutDestinationSbp $object;

    /**
     * @return PayoutDestinationSbp
     */
    protected function getTestInstance(): PayoutDestinationSbp
    {
        return new PayoutDestinationSbp();
    }

    /**
     * @return void
     */
    public function testPayoutDestinationSbpClassExists(): void
    {
        $this->object = $this->getMockBuilder(PayoutDestinationSbp::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PayoutDestinationSbp::class));
        $this->assertInstanceOf(PayoutDestinationSbp::class, $this->object);
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
        self::assertContains($instance->getType(), ['sbp']);
        self::assertContains($instance->type, ['sbp']);
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
     * Test property "phone"
     * @dataProvider validPhoneDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPhone(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPhone($value);
        self::assertNotNull($instance->getPhone());
        self::assertNotNull($instance->phone);
        self::assertEquals($value, is_array($value) ? $instance->getPhone()->toArray() : $instance->getPhone());
        self::assertEquals($value, is_array($value) ? $instance->phone->toArray() : $instance->phone);
        self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->getPhone());
        self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->phone);
    }

    /**
     * Test invalid property "phone"
     * @dataProvider invalidPhoneDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPhone(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPhone($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPhoneDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_phone'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPhoneDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_phone'));
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
     * Test property "recipient_checked"
     * @dataProvider validRecipientCheckedDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testRecipientChecked(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setRecipientChecked($value);
        self::assertNotNull($instance->getRecipientChecked());
        self::assertNotNull($instance->recipient_checked);
        self::assertEquals($value, is_array($value) ? $instance->getRecipientChecked()->toArray() : $instance->getRecipientChecked());
        self::assertEquals($value, is_array($value) ? $instance->recipient_checked->toArray() : $instance->recipient_checked);
        self::assertIsBool($instance->getRecipientChecked());
        self::assertIsBool($instance->recipient_checked);
    }

    /**
     * Test invalid property "recipient_checked"
     * @dataProvider invalidRecipientCheckedDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidRecipientChecked(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRecipientChecked($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validRecipientCheckedDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_recipient_checked'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidRecipientCheckedDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_recipient_checked'));
    }
}
