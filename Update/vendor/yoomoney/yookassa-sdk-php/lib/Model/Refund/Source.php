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

namespace YooKassa\Model\Refund;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель RefundSourcesData.
 *
 * Данные о том, с какого магазина и какую сумму нужно удержать для проведения возврата.
 * Сейчас в этом параметре можно передать данные только одного магазина.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property AmountInterface $amount Сумма возврата
 * @property AmountInterface $platformFeeAmount Комиссия, которую вы удержали при оплате, и хотите вернуть
 * @property AmountInterface $platform_fee_amount Комиссия, которую вы удержали при оплате, и хотите вернуть
 * @property string $accountId Идентификатор магазина, для которого вы хотите провести возврат
 * @property string $account_id Идентификатор магазина, для которого вы хотите провести возврат
 */
class Source extends AbstractObject implements SourceInterface
{
    /**
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_account_id = null;

    /**
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * @var AmountInterface|null
     */
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_platform_fee_amount = null;

    /**
     * Возвращает id магазина с которого будут списаны средства.
     *
     * @return string|null Id магазина с которого будут списаны средства
     */
    public function getAccountId(): ?string
    {
        return $this->_account_id;
    }

    /**
     * Устанавливает id магазина-получателя средств.
     *
     * @param string|null $account_id Id магазина с которого будут списаны средства
     *
     * @return self
     */
    public function setAccountId(?string $account_id = null): self
    {
        $this->_account_id = $this->validatePropertyValue('_account_id', $account_id);
        return $this;
    }

    /**
     * Возвращает сумму оплаты.
     *
     * @return AmountInterface|null Сумма оплаты
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает сумму оплаты.
     *
     * @param AmountInterface|array|null $amount Сумма оплаты
     *
     * @return self
     */
    public function setAmount(mixed $amount = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $amount);
        return $this;
    }

    /**
     * Возвращает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.
     *
     * @return AmountInterface|null Сумма комиссии
     */
    public function getPlatformFeeAmount(): ?AmountInterface
    {
        return $this->_platform_fee_amount;
    }

    /**
     * Устанавливает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.
     *
     * @param AmountInterface|array|null $platform_fee_amount Сумма комиссии
     *
     * @return self
     */
    public function setPlatformFeeAmount(mixed $platform_fee_amount = null): self
    {
        $this->_platform_fee_amount = $this->validatePropertyValue('_platform_fee_amount', $platform_fee_amount);
        return $this;
    }

    /**
     * Проверяет, была ли установлена сумма оплаты.
     *
     * @return bool True если сумма оплаты была установлена, false если нет
     */
    public function hasAmount(): bool
    {
        return !empty($this->_amount);
    }

    /**
     * Проверяет, была ли установлена комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу.
     *
     * @return bool True если комиссия была установлена, false если нет
     */
    public function hasPlatformFeeAmount(): bool
    {
        return !empty($this->_platform_fee_amount);
    }
}
