def get_background_prompt(unique_token, class_token):
    background = [
        'a {0} {1} in the forest.'.format(unique_token, class_token),
        'a {0} {1} in the desert.'.format(unique_token, class_token),
        'a {0} {1} on a mountaintop.'.format(unique_token, class_token),
        'a {0} {1} in a garden.'.format(unique_token, class_token),
        'a {0} {1} in a park.'.format(unique_token, class_token),
        'a {0} {1} in a meadow.'.format(unique_token, class_token),
        'a {0} {1} on a farm.'.format(unique_token, class_token),
        'a {0} {1} on a boat in the ocean.'.format(unique_token, class_token),
        'a {0} {1} on a yacht in a bay.'.format(unique_token, class_token),
        'a {0} {1} on a cruise ship in the sea.'.format(unique_token, class_token),
        'a {0} {1} on a sailboat in a lake.'.format(unique_token, class_token),
        'a {0} {1} on a river rafting trip.'.format(unique_token, class_token),
        'a {0} {1} on a horseback riding tour.'.format(unique_token, class_token),
        'a {0} {1} on a bicycle tour in the countryside.'.format(unique_token, class_token),
        'a {0} {1} on a hiking trip in the mountains.'.format(unique_token, class_token),
        'a {0} {1} on a road trip across the country.'.format(unique_token, class_token),
        'a {0} {1} in a hot air balloon over the city.'.format(unique_token, class_token),
        'a {0} {1} on a helicopter tour of a landmark.'.format(unique_token, class_token),
        'a {0} {1} in a vintage car driving down a scenic route.'.format(unique_token, class_token),
        'a {0} {1} in a cable car over the mountains.'.format(unique_token, class_token)
    ]
    return background

def get_stylization_prompt(unique_token, class_token):
    stylization = [
        'a Banksy art of {0} {1}'.format(unique_token, class_token),
        'a Vincent Van Gogh painting of {0} {1}'.format(unique_token, class_token),
        'a colorful graffiti of {0} {1}'.format(unique_token, class_token),
        'a watercolor painting of {0} {1}'.format(unique_token, class_token),
        'a Greek sculpture of {0} {1}'.format(unique_token, class_token),
        'a modern art installation of {0} {1}'.format(unique_token, class_token),
        'a Renaissance painting of {0} {1}'.format(unique_token, class_token),
        'a pop art portrait of {0} {1}'.format(unique_token, class_token),
        'a street art mural of {0} {1}'.format(unique_token, class_token),
        'a realistic portrait of {0} {1}'.format(unique_token, class_token),
        'a landscape painting of {0} {1}'.format(unique_token, class_token),
        'a black and white photograph of {0} {1}'.format(unique_token, class_token),
        'a pointillism painting of {0} {1}'.format(unique_token, class_token),
        'a still life painting of {0} {1}'.format(unique_token, class_token),
        'a surrealist painting of {0} {1}'.format(unique_token, class_token),
        'a digital art of {0} {1}'.format(unique_token, class_token),
        'an abstract expressionist painting of {0} {1}'.format(unique_token, class_token),
        'a Japanese woodblock print of {0} {1}'.format(unique_token, class_token),
        'a portrait sculpture of {0} {1}'.format(unique_token, class_token),
        'a street art stencil of {0} {1}'.format(unique_token, class_token)
    ]
    return stylization

def get_action_prompt(unique_token, class_token):
    action = [
        'a {0} {1} is writing in their journal at a coffee shop'.format(unique_token, class_token),
        'a {0} {1} is performing a magic trick for a small audience'.format(unique_token, class_token),
        'a {0} {1} is skateboarding in a skatepark'.format(unique_token, class_token),
        'a {0} {1} is practicing their dance routine in a dance studio'.format(unique_token, class_token),
        'a {0} {1} is doing a puzzle on a rainy day'.format(unique_token, class_token),
        'a {0} {1} is singing karaoke with friends at a bar'.format(unique_token, class_token),
        'a {0} {1} is building a sandcastle on the beach'.format(unique_token, class_token),
        'a {0} {1} is playing video games on a lazy afternoon'.format(unique_token, class_token),
        'a {0} {1} is having a picnic in the park with friends'.format(unique_token, class_token),
        'a {0} {1} is trying on clothes at a clothing store'.format(unique_token, class_token),
        'a {0} {1} is rock climbing at an indoor gym'.format(unique_token, class_token),
        'a {0} {1} is gardening in their backyard'.format(unique_token, class_token),
        'a {0} {1} is teaching a class of students'.format(unique_token, class_token),
        'a {0} {1} is taking a walk in the park'.format(unique_token, class_token),
        'a {0} {1} is playing frisbee with friends at the beach'.format(unique_token, class_token),
        'a {0} {1} is going for a run in the morning'.format(unique_token, class_token),
        'a {0} {1} is attending a networking event'.format(unique_token, class_token),
        'a {0} {1} is hiking in a national park'.format(unique_token, class_token),
        'a {0} {1} is attending a tech conference'.format(unique_token, class_token),
        'a {0} {1} is attending an art exhibition'.format(unique_token, class_token)
    ]
    return action 

def get_prompts(unique_token, class_token, prompt_type):
    if prompt_type == 'background':
        return get_background_prompt(unique_token, class_token)
    elif prompt_type == 'stylization':
        return get_stylization_prompt(unique_token, class_token)
    elif prompt_type == 'action':
        return get_action_prompt(unique_token, class_token)
    else:
        return get_action_prompt(unique_token, class_token) + get_background_prompt(unique_token, class_token) + get_stylization_prompt(unique_token, class_token)



if __name__ == '__main__':
    actions = get_action_prompt('sks', 'man')

    for action in actions:
        print(action)
