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

namespace YooKassa\Request\Deals;

use YooKassa\Common\AbstractRequest;
use YooKassa\Model\Deal\DealType;
use YooKassa\Model\Deal\FeeMoment;
use YooKassa\Model\Deal\SafeDeal;
use YooKassa\Model\Metadata;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель SafeDealRequest.
 *
 * Класс объекта запроса к API на проведение новой сделки.
 *
 * @example 02-builder.php 252 19 Пример использования билдера
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $type Тип сделки
 * @property string $feeMoment Момент перечисления вам вознаграждения платформы
 * @property string $fee_moment Момент перечисления вам вознаграждения платформы
 * @property string $description Описание сделки
 * @property Metadata $metadata Метаданные привязанные к сделке
 */
class CreateDealRequest extends AbstractRequest implements CreateDealRequestInterface
{
    /**
     * Тип сделки
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [DealType::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_type = DealType::SAFE_DEAL;

    /**
     * Момент перечисления вам вознаграждения платформы
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [FeeMoment::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_fee_moment = FeeMoment::PAYMENT_SUCCEEDED;

    /**
     * Описание сделки (не более 128 символов).
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: SafeDeal::MAX_LENGTH_DESCRIPTION)]
    private ?string $_description = null;

    /**
     * Метаданные привязанные к сделке.
     *
     * @var Metadata|null
     */
    #[Assert\Type(Metadata::class)]
    private ?Metadata $_metadata = null;

    /**
     * Возвращает тип сделки.
     *
     * @return string|null Тип сделки
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает тип сделки.
     *
     * @param string|null $type Тип сделки
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Проверяет наличие типа в создаваемой сделке.
     *
     * @return bool True если тип сделки есть, false если нет
     */
    public function hasType(): bool
    {
        return isset($this->_type);
    }

    /**
     * Возвращает момент перечисления вам вознаграждения платформы.
     *
     * @return string|null Момент перечисления вам вознаграждения платформы
     */
    public function getFeeMoment(): ?string
    {
        return $this->_fee_moment;
    }

    /**
     * Устанавливает момент перечисления вам вознаграждения платформы.
     *
     * @param string|null $fee_moment  Момент перечисления вам вознаграждения платформы
     *
     * @return self
     */
    public function setFeeMoment(?string $fee_moment = null): self
    {
        $this->_fee_moment = $this->validatePropertyValue('_fee_moment', $fee_moment);
        return $this;
    }

    /**
     * Проверяет, был ли установлен момент перечисления вознаграждения.
     *
     * @return bool True если момент перечисления был установлен, false если нет
     */
    public function hasFeeMoment(): bool
    {
        return !empty($this->_fee_moment);
    }

    /**
     * Возвращает описание сделки.
     *
     * @return string|null Описание сделки.
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает описание сделки.
     *
     * @param string|null $description Описание сделки (не более 128 символов)
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Проверяет наличие описания в создаваемой сделке.
     *
     * @return bool True если описание сделки есть, false если нет
     */
    public function hasDescription(): bool
    {
        return null !== $this->_description;
    }

    /**
     * Возвращает данные оплаты установленные мерчантом
     *
     * @return Metadata|null Метаданные, привязанные к платежу
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Проверяет, были ли установлены метаданные заказа.
     *
     * @return bool True если метаданные были установлены, false если нет
     */
    public function hasMetadata(): bool
    {
        return !empty($this->_metadata) && $this->_metadata->count() > 0;
    }

    /**
     * Устанавливает метаданные, привязанные к платежу.
     *
     * @param string|array|null $metadata Любые дополнительные данные, которые нужны вам для работы.
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }

    /**
     * Проверяет на валидность текущий объект
     *
     * @return bool True если объект запроса валиден, false если нет
     */
    public function validate(): bool
    {
        if (!$this->hasType()) {
            $this->setValidationError('Type field is required');

            return false;
        }
        if (!$this->hasFeeMoment()) {
            $this->setValidationError('FeeMoment field is required');

            return false;
        }

        return true;
    }

    /**
     * Возвращает билдер объектов запросов создания сделки.
     *
     * @return CreateDealRequestBuilder Инстанс билдера объектов запросов
     */
    public static function builder(): CreateDealRequestBuilder
    {
        return new CreateDealRequestBuilder();
    }
}
