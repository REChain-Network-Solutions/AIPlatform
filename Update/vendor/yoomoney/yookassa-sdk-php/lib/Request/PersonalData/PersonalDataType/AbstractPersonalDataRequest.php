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

namespace YooKassa\Request\PersonalData\PersonalDataType;

use YooKassa\Common\AbstractRequest;
use YooKassa\Model\Metadata;
use YooKassa\Model\PersonalData\PersonalDataType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PersonalDataRequest.
 *
 * Данные для запроса сохранения персональных данных.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Тип персональных данных.
 * @property Metadata $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).
*/
abstract class AbstractPersonalDataRequest extends AbstractRequest
{
    /** Максимальная длина строки фамилии или отчества */
    public const MAX_LENGTH_LAST_NAME = 200;

    /** Максимальная длина строки имени */
    public const MAX_LENGTH_FIRST_NAME = 100;

    /**
     * Тип персональных данных — цель, для которой вы будете использовать данные
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [PersonalDataType::class, 'getValidValues'])]
    protected ?string $_type = null;

    /**
     * Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).
     *
     * Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa.
     * Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @var Metadata|null
     */
    #[Assert\AllType('string')]
    #[Assert\Type(Metadata::class)]
    protected ?Metadata $_metadata = null;

    /**
     * Возвращает type.
     *
     * @return string|null
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает type.
     *
     * @param string|null $type
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Проверяет наличие типа персональных данных в запросе.
     *
     * @return bool True если тип персональных данных задан, false если нет
     */
    public function hasType(): bool
    {
        return !empty($this->_type);
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
     * Проверяет, были ли установлены метаданные.
     *
     * @return bool True если метаданные были установлены, false если нет
     */
    public function hasMetadata(): bool
    {
        return !empty($this->_metadata) && $this->_metadata->count() > 0;
    }

}
