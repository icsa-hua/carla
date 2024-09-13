import pygame
import cv2 
import warnings 
import sys 
import pdb 
import pygame_gui
from pygame_gui.core import ObjectID


class PyGUI:
    def __init__(self, image_path):
        pygame.init()

        image = cv2.imread(image_path)
        height, width, _ = image.shape
        height += 100

        pygame.display.set_caption('Clickable Map with Buttons')
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

        manager = pygame_gui.UIManager((width, height), theme_path="C:/carla/PythonAPI/carla/agents/navigation/dummy_application/dummy_utils/GUI/theme.json")

        clock = pygame.time.Clock() 
        
        map_image = pygame.image.load(image_path)
        is_running = True


        # Button settings
        # self.button_color = (95, 158, 160)
        # self.button_hover_color = (100, 100, 255)
        # self.button_font = pygame.font.SysFont(None, 25)
        # self.buttons = {
        #     'Set Start': pygame.Rect(50, 600, 175, 50),
        #     'Set Destination': pygame.Rect(250, 600, 175, 50),
        #     'Done': pygame.Rect(450, 600, 175, 50),
        # }



        
        start_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(50, 580, 175, 50),
                                             text='Set Start',
                                             manager=manager,
                                             object_id=ObjectID(class_id='button'))

        destination_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(250, 580, 175, 50),
                                             text='Set Destination',
                                             manager=manager, 
                                             object_id=ObjectID(class_id='button'))

        done_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(450, 580, 175, 50),
                                             text=' Done ',
                                             manager=manager, 
                                             object_id=ObjectID(class_id='button'))


        # Store points
        start_point = None
        destination_point = None
        selecting_start = False
        selecting_destination = False


        while is_running:

            time_delta = clock.tick(60)/1000.0

            # self.screen.fill((255, 255, 255))
            # self.screen.blit(map_image, (0, 0))
            # self.draw_buttons()


            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    is_running = False


                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == start_button: 
                            print('Selecting start point')
                            selecting_start = True
                            selecting_destination = False
                            if event.type == pygame.MOUSEBUTTONDOWN: 
                                x, y = event.pos 

                        elif event.ui_element == destination_button:
                            print('Selecting destination point')
                            selecting_start = False
                            selecting_destination = True
                            if event.type == pygame.MOUSEBUTTONDOWN: 
                                x, y = event.pos 

                        elif event.ui_element == done_button:
                            print('Done')
                            print(f"Starting Point: {start_point}")
                            print(f"Destination Point: {destination_point}")
                            selecting_start = False
                            selecting_destination = False
                            pygame.quit() 


                manager.process_events(event)

            if selecting_start : 
                pygame.draw.circle(self.screen, (0, 255, 0), start_point, 10)
            
            if selecting_destination:
                pygame.draw.circle(self.screen, (255, 0, 0), destination_point, 10)

            manager.update(time_delta)
            self.screen.blit(map_image, (0, 0))
            manager.draw_ui(self.screen)

            pygame.display.update()

            pygame.display.flip()


        if not selecting_destination and not selecting_start: 
                warnings.warn("No points selected. Please select start and destination points.")
                pygame.quit()
                sys.exit()

        pygame.quit()  


    def draw_buttons(self): 
        for name, rect in self.buttons.items():
            pygame.draw.rect(self.screen, self.button_color, rect)
            label = self.button_font.render(name, True, (255, 255, 255))
            self.screen.blit(label, (rect.x + 20, rect.y + 10))