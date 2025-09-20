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

namespace YooKassa\Model\Deal;

use DateTime;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\Metadata;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель SafeDeal.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $id Идентификатор сделки.
 * @property string $fee_moment Момент перечисления вам вознаграждения платформы.
 * @property string $feeMoment Момент перечисления вам вознаграждения платформы.
 * @property string $description Описание сделки (не более 128 символов).
 * @property string $status Статус сделки.
 * @property AmountInterface $balance Баланс сделки.
 * @property AmountInterface $payout_balance Сумма вознаграждения продавцаю.
 * @property AmountInterface $payoutBalance Сумма вознаграждения продавца.
 * @property DateTime $created_at Время создания сделки.
 * @property DateTime $createdAt Время создания сделки.
 * @property DateTime $expires_at Время автоматического закрытия сделки.
 * @property DateTime $expiresAt Время автоматического закрытия сделки.
 * @property Metadata $metadata Любые дополнительные данные, которые нужны вам для работы.
 * @property bool $test Признак тестовой операции.
 */
class SafeDeal extends AbstractBaseDeal implements DealInterface
{
    /** @var int Максимальная длина строки id сделки. */
    public const MAX_LENGTH_ID = 50;

    /** @var int Минимальная длина строки id сделки. */
    public const MIN_LENGTH_ID = 36;

    /** @var int Максимальная длина строки описания сделки. */
    public const MAX_LENGTH_DESCRIPTION = 128;

    /**
     * Идентификатор сделки.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_ID)]
    #[Assert\Length(min: self::MIN_LENGTH_ID)]
    protected ?string $_id = null;

    /**
     * Момент перечисления вам вознаграждения платформы
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [FeeMoment::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_fee_moment = null;

    /**
     * Описание сделки (не более 128 символов). Используется для фильтрации при [получении списка сделок](/developers/api#get_deals_list).
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_DESCRIPTION)]
    protected ?string $_description = null;

    /**
     * Баланс сделки
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(DealBalanceAmount::class)]
    protected ?AmountInterface $_balance = null;

    /**
     * Сумма вознаграждения продавца
     *
     * @var AmountInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(DealBalanceAmount::class)]
    protected ?AmountInterface $_payout_balance = null;

    /**
     * Статус сделки
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [DealStatus::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_status = null;

    /**
     * Время создания сделки.
     *
     * @var DateTime|null
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_created_at = null;

    /**
     * Время автоматического закрытия сделки.
     *
     * @var DateTime|null
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    protected ?DateTime $_expires_at = null;

    /**
     * Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).
     *
     * @var Metadata|null
     */
    #[Assert\Type(Metadata::class)]
    protected ?Metadata $_metadata = null;

    /**
     * Признак тестовой операции.
     *
     * @var bool|null
     */
    #[Assert\NotNull]
    #[Assert\Type('bool')]
    protected ?bool $_test = false;

    /**
     * Конструктор SafeDeal.
     *
     * @param array|null $data
     */
    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(DealType::SAFE_DEAL);
    }

    /**
     * {@inheritDoc}
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает Id сделки.
     *
     * @param string|null $id Идентификатор сделки.
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * {@inheritDoc}
     */
    public function getFeeMoment(): ?string
    {
        return $this->_fee_moment;
    }

    /**
     * Устанавливает момент перечисления вам вознаграждения платформы.
     *
     * @param string|null $fee_moment Момент перечисления вам вознаграждения платформы
     *
     * @return self
     */
    public function setFeeMoment(?string $fee_moment = null): self
    {
        $this->_fee_moment = $this->validatePropertyValue('_fee_moment', $fee_moment);
        return $this;
    }

    /**
     * {@inheritDoc}
     */
    public function getBalance(): ?AmountInterface
    {
        return $this->_balance;
    }

    /**
     * Устанавливает баланс сделки.
     *
     * @param AmountInterface|array|null $balance
     *
     * @return self
     */
    public function setBalance(mixed $balance = null): self
    {
        $this->_balance = $this->validatePropertyValue('_balance', $balance);
        return $this;
    }

    /**
     * {@inheritDoc}
     */
    public function getPayoutBalance(): ?AmountInterface
    {
        return $this->_payout_balance;
    }

    /**
     * Устанавливает сумму вознаграждения продавца.
     *
     * @param AmountInterface|array|null $payout_balance Сумма вознаграждения продавца
     *
     * @return self
     */
    public function setPayoutBalance(mixed $payout_balance = null): self
    {
        $this->_payout_balance = $this->validatePropertyValue('_payout_balance', $payout_balance);
        return $this;
    }

    /**
     * {@inheritDoc}
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Устанавливает статус сделки.
     *
     * @param string|null $status Статус сделки
     *
     * @return self
     */
    public function setStatus(?string $status = null): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * {@inheritDoc}
     */
    public function getCreatedAt(): ?DateTime
    {
        return $this->_created_at;
    }

    /**
     * Устанавливает created_at.
     *
     * @param DateTime|string|null $created_at Время создания сделки.
     *
     * @return self
     */
    public function setCreatedAt(DateTime|string|null $created_at = null): self
    {
        $this->_created_at = $this->validatePropertyValue('_created_at', $created_at);
        return $this;
    }

    /**
     * {@inheritDoc}
     */
    public function getExpiresAt(): ?DateTime
    {
        return $this->_expires_at;
    }

    /**
     * Устанавливает время автоматического закрытия сделки.
     *
     * @param DateTime|string|null $expires_at Время автоматического закрытия сделки.
     *
     * @return self
     */
    public function setExpiresAt(DateTime|string|null $expires_at = null): self
    {
        $this->_expires_at = $this->validatePropertyValue('_expires_at', $expires_at);
        return $this;
    }

    /**
     * {@inheritDoc}
     */
    public function getTest(): ?bool
    {
        return $this->_test;
    }

    /**
     * Устанавливает признак тестовой операции.
     *
     * @param bool|null $test Признак тестовой операции
     *
     * @return self
     */
    public function setTest(?bool $test = null): self
    {
        $this->_test = $this->validatePropertyValue('_test', $test);
        return $this;
    }

    /**
     * {@inheritDoc}
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает описание сделки (не более 128 символов).
     *
     * @param string|null $description Описание сделки (не более 128 символов).
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * {@inheritDoc}
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Устанавливает metadata.
     *
     * @param Metadata|array|null $metadata Любые дополнительные данные, которые нужны вам для работы.
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }
}
