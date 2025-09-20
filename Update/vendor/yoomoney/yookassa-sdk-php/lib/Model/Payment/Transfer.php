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

namespace YooKassa\Model\Payment;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель Transfer.
 *
 * Данные о распределении денег — сколько и в какой магазин нужно перевести.
 * Присутствует, если вы используете [Сплитование платежей](/developers/solutions-for-platforms/split-payments/basics).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $accountId Идентификатор магазина, в пользу которого вы принимаете оплату
 * @property string $account_id Идентификатор магазина, в пользу которого вы принимаете оплату
 * @property AmountInterface $amount Сумма, которую необходимо перечислить магазину
 * @property AmountInterface $platformFeeAmount Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу
 * @property AmountInterface $platform_fee_amount Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу
 * @property string $status Статус распределения денег между магазинами. Возможные значения: `pending`, `waiting_for_capture`, `succeeded`, `canceled`
 * @property string $description Описание транзакции, которое продавец увидит в личном кабинете ЮKassa. (например: «Заказ маркетплейса №72»)
 * @property Metadata $metadata Любые дополнительные данные, которые нужны вам для работы с платежами (например, ваш внутренний идентификатор заказа)
 * @property bool $releaseFunds Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать
 * @property bool $release_funds Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать
 * @property string $connectedAccountId Идентификатор продавца, подключенного к вашей платформе
 * @property string $connected_account_id Идентификатор продавца, подключенного к вашей платформе
 */
class Transfer extends AbstractObject implements TransferInterface
{
    /** Максимальная длина строки описания транзакции */
    public const MAX_LENGTH_DESCRIPTION = 128;

    /**
     * Идентификатор магазина, в пользу которого вы принимаете оплату
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_account_id = null;

    /**
     * Сумма, которую необходимо перечислить магазину
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_amount = null;

    /**
     * Статус распределения денег между магазинами
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [TransferStatus::class, 'getValidValues'])]
    private ?string $_status = TransferStatus::PENDING;

    /**
     * Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу
     *
     * @var AmountInterface|null
     */
    #[Assert\Valid]
    #[Assert\Type(MonetaryAmount::class)]
    private ?AmountInterface $_platform_fee_amount = null;

    /**
     * Описание транзакции (не более 128 символов), которое продавец увидит в личном кабинете ЮKassa. Например: «Заказ маркетплейса №72».
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_DESCRIPTION)]
    private ?string $_description = null;

    /**
     * Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @var Metadata|null
     */
    #[Assert\AllType('string')]
    #[Assert\Type(Metadata::class)]
    private ?Metadata $_metadata = null;

    /**
     * Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать.
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    private bool $_release_funds = false;

    /**
     * Идентификатор продавца, подключенного к вашей платформе.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_connected_account_id = null;

    /**
     * Возвращает account_id.
     *
     * @return string|null
     */
    public function getAccountId(): ?string
    {
        return $this->_account_id;
    }

    /**
     * Устанавливает account_id.
     *
     * @param string|null $value
     *
     * @return self
     */
    public function setAccountId(?string $value = null): self
    {
        $this->_account_id = $this->validatePropertyValue('_account_id', $value);
        return $this;
    }

    /**
     * Возвращает amount.
     *
     * @return AmountInterface|null
     */
    public function getAmount(): ?AmountInterface
    {
        return $this->_amount;
    }

    /**
     * Устанавливает amount.
     *
     * @param AmountInterface|array|null $value
     *
     * @return self
     */
    public function setAmount(mixed $value = null): self
    {
        $this->_amount = $this->validatePropertyValue('_amount', $value);
        return $this;
    }

    /**
     * Возвращает status.
     *
     * @return string|null
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает status.
     *
     * @param string|null $value
     *
     * @return self
     */
    public function setStatus(?string $value = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $value);
        return $this;
    }

    /**
     * Возвращает platform_fee_amount.
     *
     * @return AmountInterface|null
     */
    public function getPlatformFeeAmount(): ?AmountInterface
    {
        return $this->_platform_fee_amount;
    }

    /**
     * Устанавливает platform_fee_amount.
     *
     * @param AmountInterface|array|null $value
     *
     * @return self
     */
    public function setPlatformFeeAmount(mixed $value = null): self
    {
        $this->_platform_fee_amount = $this->validatePropertyValue('_platform_fee_amount', $value);
        return $this;
    }

    /**
     * Возвращает description.
     *
     * @return string|null
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает description.
     *
     * @param string|null $value Описание транзакции (не более 128 символов), которое продавец увидит в личном кабинете ЮKassa. Например: «Заказ маркетплейса №72».
     *
     * @return self
     */
    public function setDescription(?string $value = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $value);
        return $this;
    }

    /**
     * Возвращает metadata.
     *
     * @return Metadata|null
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Устанавливает metadata.
     *
     * @param Metadata|array|null $value Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @return self
     */
    public function setMetadata(mixed $value = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $value);
        return $this;
    }

    /**
     * Возвращает порядок перевода денег продавцам.
     *
     * @return bool|null Порядок перевода денег продавцам
     */
    public function getReleaseFunds(): ?bool
    {
        return $this->_release_funds;
    }

    /**
     * Устанавливает порядок перевода денег продавцам.
     *
     * @param bool|null $value Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать.
     *
     * @return self
     */
    public function setReleaseFunds(?bool $value = null): self
    {
        $this->_release_funds = $this->validatePropertyValue('_release_funds', $value);
        return $this;
    }

    /**
     * Возвращает идентификатор продавца.
     *
     * @return string|null Идентификатор продавца
     */
    public function getConnectedAccountId(): ?string
    {
        return $this->_connected_account_id;
    }

    /**
     * Устанавливает идентификатор продавца.
     *
     * @param string|null $value Идентификатор продавца, подключенного к вашей платформе.
     *
     * @return self
     */
    public function setConnectedAccountId(?string $value = null): self
    {
        $this->_connected_account_id = $this->validatePropertyValue('_connected_account_id', $value);
        return $this;
    }
}
